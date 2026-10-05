
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from scipy.spatial import Delaunay, QhullError, cKDTree
from torch_geometric.utils import to_undirected


@dataclass(frozen=True)
class ProximityGraph:

    edge_index: torch.Tensor
    metadata: dict[str, object]


def _pairs_to_edge_index(
    pairs: Iterable[tuple[int, int]], num_nodes: int, device: torch.device | None = None
) -> torch.Tensor:
    ordered = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in pairs if int(u) != int(v)})
    if not ordered:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(ordered, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index.to(device or torch.device("cpu"))


def _delaunay_pairs(points: np.ndarray) -> set[tuple[int, int]]:
    try:
        triangulation = Delaunay(points)
    except QhullError as exc:
        raise RuntimeError("Unable to construct a Delaunay triangulation from the 2-D points.") from exc
    pairs: set[tuple[int, int]] = set()
    for a, b, c in triangulation.simplices:
        pairs.update(
            {
                tuple(sorted((int(a), int(b)))),
                tuple(sorted((int(b), int(c)))),
                tuple(sorted((int(c), int(a)))),
            }
        )
    return pairs


def build_proximity_graphs(
    points: np.ndarray,
    *,
    constructors: Iterable[str] = ("delaunay", "gabriel", "rng", "mst"),
    device: torch.device | None = None,
) -> dict[str, ProximityGraph]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Proximity graph constructors require an [n, 2] point array.")
    if len(points) < 3:
        raise ValueError("At least three points are required.")
    requested = tuple(dict.fromkeys(name.lower() for name in constructors))
    unknown = sorted(set(requested) - {"delaunay", "gabriel", "rng", "mst"})
    if unknown:
        raise ValueError(f"Unknown proximity constructors: {', '.join(unknown)}")

    delaunay = _delaunay_pairs(points)
    tree = cKDTree(points)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    tolerance = np.finfo(np.float64).eps * scale * 64.0

    gabriel: set[tuple[int, int]] = set()
    rng: set[tuple[int, int]] = set()
    weighted_edges: list[tuple[int, int, float]] = []
    for u, v in delaunay:
        distance = float(np.linalg.norm(points[u] - points[v]))
        weighted_edges.append((u, v, distance))

        midpoint = 0.5 * (points[u] + points[v])
        radius = 0.5 * distance
        inside = tree.query_ball_point(midpoint, max(radius - tolerance, 0.0))
        if not any(w != u and w != v for w in inside):
            gabriel.add((u, v))

        near_u = set(tree.query_ball_point(points[u], max(distance - tolerance, 0.0)))
        near_v = set(tree.query_ball_point(points[v], max(distance - tolerance, 0.0)))
        if not any(w != u and w != v for w in near_u.intersection(near_v)):
            rng.add((u, v))

    graph = nx.Graph()
    graph.add_nodes_from(range(len(points)))
    graph.add_weighted_edges_from(weighted_edges)
    mst_graph = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")
    mst = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in mst_graph.edges()}

    raw = {"delaunay": delaunay, "gabriel": gabriel, "rng": rng, "mst": mst}
    result: dict[str, ProximityGraph] = {}
    for name in requested:
        pairs = raw[name]
        result[name] = ProximityGraph(
            edge_index=_pairs_to_edge_index(pairs, len(points), device),
            metadata={
                "constructor": name,
                "num_nodes": len(points),
                "undirected_edges": len(pairs),
                "density": 2.0 * len(pairs) / max(len(points) * (len(points) - 1), 1),
                "derived_from_delaunay": name != "delaunay",
            },
        )
    return result

