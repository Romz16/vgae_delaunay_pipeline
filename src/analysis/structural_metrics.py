"""Structural graph metrics for rewiring diagnostics.

The functions in this module are intentionally model-agnostic.  They operate
on a PyG ``Data`` object or an ``edge_index`` and are designed to be called
immediately after loading the original graph, before any rewiring is applied.

Implemented metrics follow the revision protocol:
- density, average degree, coefficient of variation of degree;
- edge homophily, if labels are available;
- normalized-Laplacian spectral gap lambda2;
- Cheeger-related lower/upper bounds derived from lambda2;
- approximate diameter and average shortest-path length on the largest CC;
- sampled effective resistance;
- component and isolate counts.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class StructuralMetricConfig:
    """Settings for approximate structural metrics."""

    seed: int = 42
    max_bfs_sources: int = 256
    effective_resistance_pairs: int = 128
    compute_effective_resistance: bool = True


def clean_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return unique undirected non-self-loop edges with valid node ids."""
    edge_index = edge_index.detach().cpu().long()
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.t().contiguous()
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

    src, dst = edge_index[0], edge_index[1]
    valid = (src >= 0) & (src < num_nodes) & (dst >= 0) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]
    not_loop = src != dst
    src, dst = src[not_loop], dst[not_loop]
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    return torch.unique(torch.stack([u, v], dim=0), dim=1)


def edge_index_to_networkx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """Convert a PyG edge_index to an undirected NetworkX graph."""
    clean_edges = clean_edge_index(edge_index, num_nodes)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if clean_edges.numel() > 0:
        graph.add_edges_from((int(u), int(v)) for u, v in clean_edges.t().tolist())
    return graph


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    """Build a symmetric scipy CSR adjacency matrix without self-loops."""
    clean_edges = clean_edge_index(edge_index, num_nodes)
    if clean_edges.numel() == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    source = clean_edges[0].numpy()
    target = clean_edges[1].numpy()
    row = np.concatenate([source, target])
    col = np.concatenate([target, source])
    data = np.ones(len(row), dtype=np.float64)
    adjacency = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency.tocsr()


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor | None, num_nodes: int) -> float:
    """Compute edge homophily over unique undirected edges."""
    if y is None:
        return float("nan")
    y = y.detach().cpu().long()
    clean_edges = clean_edge_index(edge_index, num_nodes)
    if clean_edges.numel() == 0 or y.numel() < num_nodes:
        return float("nan")
    src, dst = clean_edges
    return float((y[src] == y[dst]).float().mean().item())


def basic_metrics(graph: nx.Graph) -> dict[str, float | int]:
    """Compute basic size, density and degree statistics."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = np.asarray([d for _, d in graph.degree()], dtype=np.float64)
    avg_degree = float(degrees.mean()) if degrees.size else float("nan")
    degree_std = float(degrees.std(ddof=0)) if degrees.size else float("nan")
    degree_cv = float(degree_std / avg_degree) if avg_degree and avg_degree > 0 else float("nan")
    max_degree = float(degrees.max()) if degrees.size else float("nan")
    components = list(nx.connected_components(graph)) if n else []
    largest_cc = len(max(components, key=len)) if components else 0
    return {
        "num_nodes": int(n),
        "num_edges": int(m),
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "avg_degree": avg_degree,
        "degree_std": degree_std,
        "degree_cv": degree_cv,
        "max_degree": max_degree,
        "num_components": int(len(components)),
        "largest_cc_ratio": float(largest_cc / n) if n else float("nan"),
        "num_isolates": int(nx.number_of_isolates(graph)),
    }


def spectral_metrics(edge_index: torch.Tensor, num_nodes: int) -> dict[str, float | int]:
    """Compute normalized-Laplacian lambda2 and Cheeger-related proxies."""
    adjacency = edge_index_to_adjacency(edge_index, num_nodes)
    if num_nodes < 3 or adjacency.nnz == 0:
        return {
            "lambda2_norm_laplacian": float("nan"),
            "cheeger_lower_bound": float("nan"),
            "cheeger_upper_bound": float("nan"),
            "num_isolated_nodes": int(num_nodes),
        }

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    num_isolated = int((degree == 0).sum())
    safe_degree = degree.copy()
    safe_degree[safe_degree == 0] = 1.0
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(safe_degree))
    laplacian = sp.eye(num_nodes, format="csr") - d_inv_sqrt @ adjacency @ d_inv_sqrt

    try:
        values = spla.eigsh(
            laplacian,
            k=min(3, num_nodes - 1),
            which="SM",
            return_eigenvectors=False,
            tol=1e-4,
        )
        values = np.sort(np.real(values))
        lambda2 = float(values[1]) if len(values) > 1 else float("nan")
        if np.isfinite(lambda2) and abs(lambda2) < 1e-10:
            lambda2 = 0.0
    except Exception:
        lambda2 = float("nan")

    return {
        "lambda2_norm_laplacian": lambda2,
        "cheeger_lower_bound": float(lambda2 / 2.0) if np.isfinite(lambda2) else float("nan"),
        "cheeger_upper_bound": float(math.sqrt(2.0 * lambda2)) if np.isfinite(lambda2) and lambda2 >= 0 else float("nan"),
        "num_isolated_nodes": num_isolated,
    }


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a graph."""
    components = list(nx.connected_components(graph))
    if not components:
        return graph.copy()
    return graph.subgraph(max(components, key=len)).copy()


def distance_metrics(graph: nx.Graph, max_sources: int, seed: int) -> dict[str, float]:
    """Approximate diameter and average shortest-path length on largest CC."""
    rng = random.Random(seed)
    component = largest_connected_component(graph)
    nodes = list(component.nodes())
    if len(nodes) <= 1:
        return {"approx_diameter": float("nan"), "approx_avg_shortest_path": float("nan")}

    sources = rng.sample(nodes, max_sources) if len(nodes) > max_sources else nodes
    distances: list[int] = []
    max_distance = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(component, source)
        values = [d for node, d in lengths.items() if node != source]
        if values:
            distances.extend(values)
            max_distance = max(max_distance, max(values))

    return {
        "approx_diameter": float(max_distance),
        "approx_avg_shortest_path": float(np.mean(distances)) if distances else float("nan"),
    }


def effective_resistance_sample(edge_index: torch.Tensor, num_nodes: int, num_pairs: int, seed: int) -> dict[str, float | int]:
    """Approximate average effective resistance over sampled node pairs.

    The Laplacian is grounded by removing one non-isolated node.  This is an
    exploratory approximation intended for cross-dataset comparison, not an
    exact all-pairs effective-resistance computation.
    """
    rng = np.random.default_rng(seed)
    adjacency = edge_index_to_adjacency(edge_index, num_nodes)
    if num_nodes < 3 or adjacency.nnz == 0:
        return {
            "effective_resistance_sample": float("nan"),
            "effective_resistance_median": float("nan"),
            "effective_resistance_pairs": 0,
        }

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    non_isolated = np.where(degree > 0)[0]
    if len(non_isolated) < 2:
        return {
            "effective_resistance_sample": float("nan"),
            "effective_resistance_median": float("nan"),
            "effective_resistance_pairs": 0,
        }

    laplacian = sp.diags(degree) - adjacency
    ground = int(non_isolated[-1])
    keep = np.asarray([i for i in range(num_nodes) if i != ground], dtype=np.int64)
    node_to_reduced = {int(node): idx for idx, node in enumerate(keep)}
    reduced_laplacian = laplacian[keep][:, keep].tocsc()

    values: list[float] = []
    for _ in range(num_pairs):
        u, v = rng.choice(non_isolated, size=2, replace=False)
        u, v = int(u), int(v)
        b = np.zeros(num_nodes - 1, dtype=np.float64)
        if u != ground:
            b[node_to_reduced[u]] = 1.0
        if v != ground:
            b[node_to_reduced[v]] -= 1.0
        try:
            x, info = spla.cg(reduced_laplacian, b, maxiter=1000, rtol=1e-6)
            if info == 0:
                resistance = float(b @ x)
                if np.isfinite(resistance) and resistance >= 0:
                    values.append(resistance)
        except Exception:
            continue

    return {
        "effective_resistance_sample": float(np.mean(values)) if values else float("nan"),
        "effective_resistance_median": float(np.median(values)) if values else float("nan"),
        "effective_resistance_pairs": int(len(values)),
    }


def compute_structural_metrics(
    data: Data,
    dataset: str,
    graph_name: str = "Original",
    config: StructuralMetricConfig | None = None,
    edge_index: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compute all structural metrics for a graph condition."""
    cfg = config or StructuralMetricConfig()
    selected_edge_index = edge_index if edge_index is not None else data.edge_index
    selected_edge_index = selected_edge_index.detach().cpu().long()
    y = getattr(data, "y", None)
    y = y.detach().cpu().long() if y is not None else None
    num_nodes = int(getattr(data, "num_nodes", 0) or int(selected_edge_index.max().item() + 1))
    graph = edge_index_to_networkx(selected_edge_index, num_nodes)

    row: dict[str, object] = {
        "dataset": dataset,
        "graph_name": graph_name,
        "metric_seed": cfg.seed,
        "max_bfs_sources": cfg.max_bfs_sources,
        "effective_resistance_pairs_requested": cfg.effective_resistance_pairs,
    }
    row.update(basic_metrics(graph))
    row["edge_homophily"] = edge_homophily(selected_edge_index, y, num_nodes)
    row.update(spectral_metrics(selected_edge_index, num_nodes))
    row.update(distance_metrics(graph, max_sources=cfg.max_bfs_sources, seed=cfg.seed))
    if cfg.compute_effective_resistance:
        row.update(
            effective_resistance_sample(
                selected_edge_index,
                num_nodes,
                num_pairs=cfg.effective_resistance_pairs,
                seed=cfg.seed,
            )
        )
    else:
        row.update(
            {
                "effective_resistance_sample": float("nan"),
                "effective_resistance_median": float("nan"),
                "effective_resistance_pairs": 0,
            }
        )
    return row


def save_structural_metrics(rows: list[dict[str, object]], path) -> pd.DataFrame:
    """Save structural metric rows to CSV."""
    df = pd.DataFrame(rows)
    path = path if hasattr(path, "parent") else __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def config_to_dict(config: StructuralMetricConfig) -> dict[str, object]:
    """Return a JSON-serializable structural metric config."""
    return asdict(config)
