"""Controlled perturbations for empirical graph experiments."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops, to_undirected


@dataclass(frozen=True)
class CorruptedGraph:
    """A perturbed graph and diagnostics relative to the clean original."""

    edge_index: torch.Tensor
    metadata: dict[str, object]


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    cleaned = to_undirected(remove_self_loops(edge_index.detach().cpu())[0])
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in cleaned.t().tolist() if u != v}


def _edge_index(pairs: set[tuple[int, int]], num_nodes: int, device: torch.device) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    tensor = torch.tensor(sorted(pairs), dtype=torch.long).t().contiguous()
    return to_undirected(tensor, num_nodes=num_nodes).to(device)


def _homophily(pairs: set[tuple[int, int]], labels: np.ndarray) -> float:
    if not pairs:
        return float("nan")
    return float(np.mean([labels[u] == labels[v] for u, v in pairs]))


def corrupt_graph(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_nodes: int,
    mode: str,
    fraction: float,
    seed: int,
    device: torch.device,
) -> CorruptedGraph:
    """Perturb an empirical graph while keeping its nodes and edge budget fixed."""
    mode = mode.lower()
    if mode not in {"none", "degree_preserving", "homophily_attack"}:
        raise ValueError(f"Unknown corruption mode: {mode}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Corruption fraction must be in [0, 1].")
    clean = _pairs(edge_index)
    observed = set(clean)
    label_array = labels.detach().cpu().numpy()
    rng = np.random.default_rng(seed)

    if mode == "degree_preserving" and fraction > 0.0:
        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(observed)
        swaps = max(1, int(round(fraction * len(observed))))
        try:
            nx.double_edge_swap(
                graph,
                nswap=swaps,
                max_tries=max(1000, swaps * 100),
                seed=seed,
            )
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            pass
        observed = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges()}
    elif mode == "homophily_attack" and fraction > 0.0:
        same = [(u, v) for u, v in observed if label_array[u] == label_array[v]]
        requested = min(len(same), int(round(fraction * len(observed))))
        if requested:
            chosen = rng.choice(len(same), size=requested, replace=False)
            for index in np.atleast_1d(chosen):
                observed.remove(same[int(index)])
            classes = np.unique(label_array)
            added = 0
            attempts = 0
            max_attempts = max(10000, requested * 1000)
            while added < requested and attempts < max_attempts:
                first, second = rng.choice(classes, size=2, replace=False)
                u = int(rng.choice(np.flatnonzero(label_array == first)))
                v = int(rng.choice(np.flatnonzero(label_array == second)))
                pair = (min(u, v), max(u, v))
                attempts += 1
                if u != v and pair not in observed:
                    observed.add(pair)
                    added += 1
            if added != requested:
                raise RuntimeError(f"Homophily attack added {added}/{requested} replacement edges.")

    union = clean | observed
    changed_fraction = len(clean.symmetric_difference(observed)) / max(2 * len(clean), 1)
    clean_graph = nx.Graph(); clean_graph.add_nodes_from(range(num_nodes)); clean_graph.add_edges_from(clean)
    observed_graph = nx.Graph(); observed_graph.add_nodes_from(range(num_nodes)); observed_graph.add_edges_from(observed)
    metadata = {
        "corruption_mode": mode,
        "corruption_fraction_requested": float(fraction),
        "clean_edges": len(clean),
        "observed_edges": len(observed),
        "clean_observed_edge_jaccard": len(clean & observed) / max(len(union), 1),
        "actual_changed_edge_fraction": changed_fraction,
        "clean_homophily": _homophily(clean, label_array),
        "observed_homophily": _homophily(observed, label_array),
        "degree_sequence_preserved": sorted(dict(clean_graph.degree()).values()) == sorted(dict(observed_graph.degree()).values()),
    }
    return CorruptedGraph(_edge_index(observed, num_nodes, device), metadata)

