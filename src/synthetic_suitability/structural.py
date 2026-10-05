"""Label-free structural metrics for the prerewiring indicator."""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .config import MetricStudyConfig


INDICATOR_FORBIDDEN_COLUMNS = {
    "edge_homophily", "baseline_f1", "baseline_test_f1", "gain_f1", "test_gain",
    "selected_test_f1", "gain_geometry", "gain_vgae", "success", "success_05", "success_10",
}


def compute_prerewiring_metrics(graph: nx.Graph, cfg: MetricStudyConfig, seed: int) -> dict[str, float | int]:
    """Compute metrics using only the original unlabeled topology."""
    graph = nx.convert_node_labels_to_integers(nx.Graph(graph), ordering="sorted")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    n = graph.number_of_nodes(); m = graph.number_of_edges()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    components = list(nx.connected_components(graph)) if n else []
    largest_nodes = max(components, key=len) if components else set()
    largest = graph.subgraph(largest_nodes).copy()
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), dtype=float, format="csr") if n else sparse.csr_matrix((0, 0))

    row: dict[str, float | int] = {
        "num_nodes": n,
        "num_edges": m,
        "density": float(nx.density(graph)) if n > 1 else 0.0,
        "avg_degree": float(degrees.mean()) if degrees.size else float("nan"),
        "degree_variance": float(degrees.var()) if degrees.size else float("nan"),
        "degree_std": float(degrees.std()) if degrees.size else float("nan"),
        "degree_cv": float(degrees.std() / degrees.mean()) if degrees.size and degrees.mean() > 0 else float("nan"),
        "num_components": len(components),
        "largest_cc_ratio": float(len(largest_nodes) / n) if n else float("nan"),
        "num_isolates": int(nx.number_of_isolates(graph)),
        "avg_clustering": float(nx.average_clustering(graph)) if n else float("nan"),
        "transitivity": float(nx.transitivity(graph)) if n else float("nan"),
        "degree_assortativity": _safe_assortativity(graph),
    }
    row.update(_spectral_metrics(adjacency, cfg.spectral_tolerance))
    row.update(_distance_metrics(largest, cfg.bfs_sources, seed))
    row.update(_resistance_metrics(largest, cfg.resistance_pairs, seed))
    row.update(_cut_metrics(graph, largest, adjacency, cfg, seed))
    row["modularity"] = _modularity(graph, cfg.modularity_max_nodes)
    assert not INDICATOR_FORBIDDEN_COLUMNS.intersection(row), "Forbidden leakage feature was generated."
    return row


def _safe_assortativity(graph: nx.Graph) -> float:
    try:
        value = float(nx.degree_assortativity_coefficient(graph))
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _spectral_metrics(adjacency, tolerance: float) -> dict[str, float]:
    n = adjacency.shape[0]
    if n < 3 or adjacency.nnz == 0:
        return {"lambda2_norm_laplacian": float("nan"), "algebraic_connectivity": float("nan"), "cheeger_lower_bound": float("nan"), "cheeger_upper_bound": float("nan")}
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    safe = degree.copy(); safe[safe == 0] = 1.0
    norm_lap = sparse.eye(n, format="csr") - sparse.diags(1.0 / np.sqrt(safe)) @ adjacency @ sparse.diags(1.0 / np.sqrt(safe))
    lap = sparse.diags(degree) - adjacency
    try:
        vals = np.sort(np.real(eigsh(norm_lap, k=min(3, n - 1), which="SM", return_eigenvectors=False, tol=tolerance)))
        lambda2 = float(vals[1]) if len(vals) > 1 else float("nan")
        if abs(lambda2) < 1e-10: lambda2 = 0.0
    except Exception:
        lambda2 = float("nan")
    try:
        vals = np.sort(np.real(eigsh(lap, k=min(3, n - 1), which="SM", return_eigenvectors=False, tol=tolerance)))
        algebraic = float(vals[1]) if len(vals) > 1 else float("nan")
        if abs(algebraic) < 1e-10: algebraic = 0.0
    except Exception:
        algebraic = float("nan")
    return {
        "lambda2_norm_laplacian": lambda2,
        "algebraic_connectivity": algebraic,
        "cheeger_lower_bound": lambda2 / 2.0 if np.isfinite(lambda2) else float("nan"),
        "cheeger_upper_bound": math.sqrt(2.0 * lambda2) if np.isfinite(lambda2) and lambda2 >= 0 else float("nan"),
    }


def _distance_metrics(component: nx.Graph, max_sources: int, seed: int) -> dict[str, float]:
    nodes = list(component.nodes())
    if len(nodes) <= 1:
        return {"approx_avg_shortest_path": float("nan"), "approx_diameter": float("nan")}
    rng = random.Random(seed)
    sources = rng.sample(nodes, min(len(nodes), max_sources))
    values: list[int] = []; diameter = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(component, source)
        observed = [value for node, value in lengths.items() if node != source]
        values.extend(observed)
        if observed: diameter = max(diameter, max(observed))
    return {"approx_avg_shortest_path": float(np.mean(values)), "approx_diameter": float(diameter)}


def _resistance_metrics(component: nx.Graph, pairs: int, seed: int) -> dict[str, float | int]:
    n = component.number_of_nodes()
    if n < 3 or component.number_of_edges() == 0:
        return {"effective_resistance_sample": float("nan"), "effective_resistance_median": float("nan"), "effective_resistance_pairs": 0}
    nodes = sorted(component.nodes())
    adjacency = nx.to_scipy_sparse_array(component, nodelist=nodes, dtype=float, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    lap = sparse.diags(degree) - adjacency
    ground = n - 1
    reduced = lap[:-1, :-1].tocsc()
    try:
        solve = sparse.linalg.factorized(reduced)
    except Exception:
        return {"effective_resistance_sample": float("nan"), "effective_resistance_median": float("nan"), "effective_resistance_pairs": 0}
    rng = np.random.default_rng(seed); values=[]
    for _ in range(pairs):
        u, v = map(int, rng.choice(n, size=2, replace=False))
        b = np.zeros(n - 1)
        if u != ground: b[u] += 1.0
        if v != ground: b[v] -= 1.0
        try:
            value = float(b @ solve(b))
            if np.isfinite(value) and value >= 0: values.append(value)
        except Exception:
            pass
    return {
        "effective_resistance_sample": float(np.mean(values)) if values else float("nan"),
        "effective_resistance_median": float(np.median(values)) if values else float("nan"),
        "effective_resistance_pairs": len(values),
    }


def _cut_metrics(graph: nx.Graph, largest: nx.Graph, adjacency, cfg: MetricStudyConfig, seed: int) -> dict[str, float]:
    if largest.number_of_nodes() <= 1:
        return {"edge_connectivity": 0.0, "edge_connectivity_ratio": 0.0, "spectral_sweep_conductance": float("nan")}
    try:
        if largest.number_of_nodes() <= cfg.exact_edge_connectivity_max_nodes:
            connectivity = float(nx.edge_connectivity(largest))
        else:
            rng = np.random.default_rng(seed); nodes=np.asarray(list(largest.nodes()))
            estimates=[]
            for _ in range(min(24, len(nodes))):
                u,v=map(int,rng.choice(nodes,size=2,replace=False))
                estimates.append(float(nx.local_edge_connectivity(largest,u,v,cutoff=32)))
            connectivity = min(estimates) if estimates else float("nan")
    except Exception:
        connectivity = float("nan")
    degree_sum = max(1.0, float(sum(dict(largest.degree()).values())))
    conductance = _spectral_sweep_conductance(largest, cfg.spectral_tolerance)
    return {"edge_connectivity": connectivity, "edge_connectivity_ratio": connectivity / degree_sum if np.isfinite(connectivity) else float("nan"), "spectral_sweep_conductance": conductance}


def _spectral_sweep_conductance(graph: nx.Graph, tolerance: float) -> float:
    n = graph.number_of_nodes()
    if n < 3 or not nx.is_connected(graph): return 0.0 if n >= 2 else float("nan")
    nodes=sorted(graph.nodes()); adjacency=nx.to_scipy_sparse_array(graph,nodelist=nodes,dtype=float,format="csr")
    degree=np.asarray(adjacency.sum(axis=1)).ravel(); safe=degree.copy(); safe[safe==0]=1
    lap=sparse.eye(n,format="csr")-sparse.diags(1/np.sqrt(safe))@adjacency@sparse.diags(1/np.sqrt(safe))
    try:
        _, vecs=eigsh(lap,k=2,which="SM",tol=tolerance); order=np.argsort(vecs[:,1]); total=float(degree.sum()); best=float("inf"); in_set=np.zeros(n,dtype=bool)
        for idx in order[:-1]:
            in_set[idx]=True; vol=float(degree[in_set].sum()); other=total-vol
            if min(vol,other)<=0: continue
            cut=float(adjacency[in_set][:,~in_set].sum()); best=min(best,cut/min(vol,other))
        return best if np.isfinite(best) else float("nan")
    except Exception:
        return float("nan")


def _modularity(graph: nx.Graph, max_nodes: int) -> float:
    if graph.number_of_edges()==0 or graph.number_of_nodes()>max_nodes: return float("nan")
    try:
        communities=list(nx.algorithms.community.greedy_modularity_communities(graph))
        return float(nx.algorithms.community.modularity(graph,communities))
    except Exception:
        return float("nan")
