from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from src.data.graph_corruption import corrupt_graph
from src.rewiring.proximity_graphs import build_proximity_graphs


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edge_index.t().tolist() if u != v}


def test_proximity_graph_subset_chain_and_mst_connectivity() -> None:
    points = np.random.default_rng(42).normal(size=(30, 2))
    graphs = build_proximity_graphs(points)
    pairs = {name: _pairs(result.edge_index) for name, result in graphs.items()}
    assert pairs["mst"] <= pairs["rng"] <= pairs["gabriel"] <= pairs["delaunay"]
    mst = nx.Graph(); mst.add_nodes_from(range(len(points))); mst.add_edges_from(pairs["mst"])
    assert nx.is_tree(mst)
    assert len(pairs["mst"]) == len(points) - 1


def test_controlled_corruptions_keep_edge_budget() -> None:
    graph = nx.cycle_graph(20)
    graph.add_edges_from((i, i + 2) for i in range(18))
    edges = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    labels = torch.tensor([0] * 10 + [1] * 10)
    clean_budget = graph.number_of_edges()
    for mode in ("degree_preserving", "homophily_attack"):
        result = corrupt_graph(
            edges,
            labels,
            num_nodes=20,
            mode=mode,
            fraction=0.25,
            seed=7,
            device=torch.device("cpu"),
        )
        assert len(_pairs(result.edge_index)) == clean_budget
        assert result.metadata["actual_changed_edge_fraction"] > 0.0
    preserved = corrupt_graph(
        edges,
        labels,
        num_nodes=20,
        mode="degree_preserving",
        fraction=0.25,
        seed=7,
        device=torch.device("cpu"),
    )
    assert preserved.metadata["degree_sequence_preserved"] is True

