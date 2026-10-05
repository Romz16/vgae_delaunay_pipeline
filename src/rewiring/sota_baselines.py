
from __future__ import annotations

import heapq
import math
import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops, to_undirected

from config import RewiringConfig
from src.rewiring.hybrid_vgae_ct import EffectiveResistanceScorer


def _canonical_edges(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    edge_index = to_undirected(remove_self_loops(edge_index.detach().cpu())[0])
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edge_index.t().tolist() if u != v}


def _edge_index(graph: nx.Graph) -> torch.Tensor:
    pairs = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in graph.edges() if u != v)
    rows = [node for u, v in pairs for node in (u, v)]
    cols = [node for u, v in pairs for node in (v, u)]
    return torch.tensor([rows, cols], dtype=torch.long)


def balanced_forman(u: int, v: int, graph: nx.Graph) -> float:
    di, dj = graph.degree(u), graph.degree(v)
    if di <= 1 or dj <= 1:
        return 0.0
    ni, nj = set(graph.neighbors(u)), set(graph.neighbors(v))
    triangles = ni & nj
    ni_only = ni.difference(nj).difference({v})
    nj_only = nj.difference(ni).difference({u})
    squares = {(a, b) for a in ni_only for b in graph.neighbors(a) if b in nj_only}
    square_i = {a for a, _ in squares}
    square_j = {b for _, b in squares}
    gamma_max = 0
    for node in square_i:
        gamma_max = max(gamma_max, sum(1 for w in graph.neighbors(node) if w in nj and w not in ni) - 1)
    for node in square_j:
        gamma_max = max(gamma_max, sum(1 for w in graph.neighbors(node) if w in ni and w not in nj) - 1)
    triangle_term = 2 * len(triangles) / max(di, dj) + len(triangles) / min(di, dj)
    square_term = 0.0 if gamma_max <= 0 else (len(square_i) + len(square_j)) / (gamma_max * max(di, dj))
    return 2 / di + 2 / dj - 2 + triangle_term + square_term


def _affected_edges(graph: nx.Graph, nodes: set[int]) -> set[tuple[int, int]]:
    expanded = set(nodes)
    for node in list(nodes):
        expanded.update(graph.neighbors(node))
    return {(min(int(u), int(v)), max(int(u), int(v))) for u in expanded for v in graph.neighbors(u) if u != v}


def sdrf_snapshots(
    edge_index: torch.Tensor,
    num_nodes: int,
    step_ratios: tuple[float, ...],
    *,
    seed: int,
    temperature: float = 5.0,
    removal_bound: float | None = None,
) -> dict[float, torch.Tensor]:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(_canonical_edges(edge_index))
    targets = {ratio: int(round(float(ratio) * num_nodes)) for ratio in step_ratios}
    snapshots: dict[float, torch.Tensor] = {}
    if 0.0 in targets:
        snapshots[0.0] = _edge_index(graph)
    max_steps = max(targets.values(), default=0)

    versions: dict[tuple[int, int], int] = {}
    heap: list[tuple[float, int, int, int]] = []
    for u, v in graph.edges():
        edge = (min(u, v), max(u, v))
        versions[edge] = 0
        heapq.heappush(heap, (balanced_forman(*edge, graph), edge[0], edge[1], 0))

    def refresh(edges: set[tuple[int, int]]) -> None:
        for edge in edges:
            if not graph.has_edge(*edge):
                versions.pop(edge, None)
                continue
            version = versions.get(edge, -1) + 1
            versions[edge] = version
            heapq.heappush(heap, (balanced_forman(*edge, graph), edge[0], edge[1], version))

    completed = 0
    while completed < max_steps and heap:
        while heap:
            ric, u, v, version = heapq.heappop(heap)
            edge = (u, v)
            if graph.has_edge(u, v) and versions.get(edge) == version:
                break
        else:
            break
        candidates: list[tuple[int, int]] = []
        improvements: list[float] = []
        for a in list(graph.neighbors(u)):
            for b in list(graph.neighbors(v)):
                x, y = sorted((int(a), int(b)))
                if x == y or graph.has_edge(x, y) or (x, y) in candidates:
                    continue
                graph.add_edge(x, y)
                improvements.append(balanced_forman(u, v, graph) - ric)
                candidates.append((x, y))
                graph.remove_edge(x, y)
        if not candidates:
            versions[edge] = version + 1
            continue
        scores = np.asarray(improvements, dtype=np.float64) * temperature
        scores -= np.max(scores)
        probabilities = np.exp(np.clip(scores, -700, 700))
        probabilities /= probabilities.sum()
        added = candidates[int(rng.choice(len(candidates), p=probabilities))]
        graph.add_edge(*added)
        changed_nodes = {u, v, *added}

        if removal_bound is not None:
            local_edges = _affected_edges(graph, changed_nodes)
            if local_edges:
                remove_edge = max(local_edges, key=lambda item: balanced_forman(*item, graph))
                if balanced_forman(*remove_edge, graph) > removal_bound and remove_edge != added:
                    graph.remove_edge(*remove_edge)
                    changed_nodes.update(remove_edge)
        refresh(_affected_edges(graph, changed_nodes))
        completed += 1
        for ratio, target in targets.items():
            if ratio not in snapshots and completed >= target:
                snapshots[ratio] = _edge_index(graph)

    for ratio in step_ratios:
        snapshots.setdefault(ratio, _edge_index(graph))
    return snapshots


def diffwire_ct_edge_weights(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: RewiringConfig,
) -> tuple[torch.Tensor, dict[str, object]]:
    clean = to_undirected(remove_self_loops(edge_index.detach().cpu())[0], num_nodes=num_nodes)
    model = EffectiveResistanceScorer(config).fit(clean, num_nodes)
    pairs = clean.t().numpy().astype(np.int64)
    resistance = model.resistance(pairs)
    finite = resistance[np.isfinite(resistance)]
    scale = float(np.median(finite)) if finite.size else 1.0
    weights = resistance / max(scale, config.ct_eps)
    weights = np.clip(weights, config.ct_eps, 100.0)
    return torch.as_tensor(weights, dtype=torch.float32), {
        "implementation": "DiffWire CT spectral CTE edge relevance",
        "official_reference_commit": "deced4bbe088827e39a9359fa368a8efa2b00cfd",
        "num_eigenvectors": int(model.coordinates.shape[1]),
        "normalization": "effective_resistance / median_effective_resistance",
        "dense_layer_replaced_for_scalability": True,
    }
