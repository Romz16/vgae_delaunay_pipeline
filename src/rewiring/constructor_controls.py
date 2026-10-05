
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import remove_self_loops, to_undirected


@dataclass(frozen=True)
class ConstructorResult:
    edge_index: torch.Tensor
    metadata: dict[str, object]


def _pairs_from_edge_index(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    edge_index = to_undirected(remove_self_loops(edge_index.detach().cpu())[0])
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edge_index.t().tolist() if u != v}


def _edge_index(pairs: Iterable[tuple[int, int]], num_nodes: int) -> torch.Tensor:
    ordered = sorted(set(pairs))
    rows: list[int] = []
    cols: list[int] = []
    for u, v in ordered:
        rows.extend((u, v))
        cols.extend((v, u))
    if not ordered:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def _neighbor_candidates(points: np.ndarray, k: int) -> list[tuple[float, int, int, bool]]:
    n = len(points)
    k = min(max(1, int(k)), n - 1)
    distances, indices = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(points).kneighbors(points)
    directed = {(i, int(j)): float(d) for i, (row_i, row_d) in enumerate(zip(indices, distances)) for j, d in zip(row_i[1:], row_d[1:])}
    canonical: dict[tuple[int, int], tuple[float, bool]] = {}
    for (i, j), distance in directed.items():
        u, v = sorted((i, j))
        reverse = (j, i) in directed
        current = canonical.get((u, v))
        best_distance = min(distance, current[0]) if current is not None else distance
        canonical[(u, v)] = (best_distance, reverse or (current[1] if current is not None else False))
    return [(distance, u, v, reverse) for (u, v), (distance, reverse) in canonical.items()]


def _budgeted(rows: list[tuple[float, int, int, bool]], budget: int) -> set[tuple[int, int]]:
    if len(rows) < budget:
        raise RuntimeError(f"Candidate pool has {len(rows)} edges, below requested budget {budget}.")
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return {(u, v) for _, u, v, _ in rows[:budget]}


def _pool_for_budget(points: np.ndarray, budget: int, mutual: bool = False) -> tuple[list[tuple[float, int, int, bool]], int]:
    n = len(points)
    k = min(max(8, int(np.ceil(2 * budget / max(n, 1)))), n - 1)
    while True:
        rows = _neighbor_candidates(points, k)
        if mutual:
            rows = [row for row in rows if row[3]]
        if len(rows) >= budget or k >= n - 1:
            return rows, k
        k = min(n - 1, max(k + 1, int(np.ceil(k * 1.6))))


def build_constructor_controls(
    points: np.ndarray,
    delaunay_edge_index: torch.Tensor,
    *,
    seed: int,
) -> dict[str, ConstructorResult]:
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    delaunay_pairs = _pairs_from_edge_index(delaunay_edge_index)
    budget = len(delaunay_pairs)
    if n < 2 or budget < 1:
        raise ValueError("At least two nodes and one Delaunay edge are required.")

    knn_rows, knn_k = _pool_for_budget(points, budget, mutual=False)
    mutual_rows, mutual_k = _pool_for_budget(points, budget, mutual=True)
    knn_pairs = _budgeted(knn_rows, budget)
    mutual_pairs = _budgeted(mutual_rows, budget)

    # An exact-budget radius graph is the prefix of globally shortest local
    # pairs.  The cutoff is stored so the construction remains reproducible.
    radius_pairs = _budgeted(list(knn_rows), budget)
    radius_lookup = {(u, v): d for d, u, v, _ in knn_rows}
    radius = max(radius_lookup[pair] for pair in radius_pairs)

    # Euclidean MST (computed on the sufficiently rich kNN pool) plus shortest
    # remaining kNN edges until it reaches the Delaunay edge budget.
    mst_rows, mst_cols, mst_data = [], [], []
    for distance, u, v, _ in knn_rows:
        mst_rows.extend((u, v))
        mst_cols.extend((v, u))
        mst_data.extend((distance, distance))
    sparse_dist = coo_matrix((mst_data, (mst_rows, mst_cols)), shape=(n, n)).tocsr()
    mst = minimum_spanning_tree(sparse_dist).tocoo()
    mst_pairs = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(mst.row, mst.col)}
    initial_mst_edges = len(mst_pairs)
    for _, u, v, _ in sorted(knn_rows):
        if len(mst_pairs) >= budget:
            break
        mst_pairs.add((u, v))
    if len(mst_pairs) != budget:
        raise RuntimeError(f"MST+kNN produced {len(mst_pairs)} edges, expected {budget}.")

    rng = np.random.default_rng(seed)
    random_pairs: set[tuple[int, int]] = set()
    while len(random_pairs) < budget:
        needed = budget - len(random_pairs)
        uv = rng.integers(0, n, size=(max(needed * 2, 1024), 2))
        for u_raw, v_raw in uv:
            u, v = sorted((int(u_raw), int(v_raw)))
            if u != v:
                random_pairs.add((u, v))
            if len(random_pairs) == budget:
                break

    randomized = nx.Graph()
    randomized.add_nodes_from(range(n))
    randomized.add_edges_from(delaunay_pairs)
    requested_swaps = max(1, 10 * budget)
    completed_swaps = requested_swaps
    try:
        nx.double_edge_swap(randomized, nswap=requested_swaps, max_tries=max(100, 100 * budget), seed=seed)
    except (nx.NetworkXAlgorithmError, nx.NetworkXError):
        completed_swaps = 0
    degree_pairs = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in randomized.edges()}

    raw = {
        "knn": (knn_pairs, {"k_search": knn_k}),
        "mutual_knn": (mutual_pairs, {"k_search": mutual_k}),
        "random_edge_budget_matched": (random_pairs, {"seed": seed}),
        "random_degree_matched": (degree_pairs, {"seed": seed, "requested_double_edge_swaps": requested_swaps, "completed": completed_swaps > 0}),
        "radius": (radius_pairs, {"radius": radius}),
        "mst_knn": (mst_pairs, {"mst_edges": initial_mst_edges, "k_search": knn_k}),
    }
    results: dict[str, ConstructorResult] = {}
    for name, (pairs, extra) in raw.items():
        if len(pairs) != budget:
            raise RuntimeError(f"{name} has {len(pairs)} edges; expected Delaunay budget {budget}.")
        results[name] = ConstructorResult(
            edge_index=_edge_index(pairs, n),
            metadata={"constructor": name, "num_nodes": n, "undirected_edges": budget, "delaunay_budget": budget, **extra},
        )
    return results
