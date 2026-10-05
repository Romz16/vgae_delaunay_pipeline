"""VGAE-guided rewiring over the DGlf Delaunay graph."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import add_self_loops, to_undirected

LOGGER = logging.getLogger("vgae_delaunay_pipeline.vgae_delaunay_rewiring")


class VGAEDelaunayRewirer:
    """Apply percentage-based VGAE rewiring to a Delaunay candidate graph.

    This reproduces the experimental idea used previously: the base graph for
    the rewired condition is the DGlf Delaunay graph. A ratio of 0% therefore
    means a pure DGlf/Delaunay graph, not the original graph baseline.
    """

    def __init__(self, add_self_loops: bool = True, device: torch.device | None = None) -> None:
        """Initialize the rewirer.

        Args:
            add_self_loops: Whether to add self-loops after rewiring.
            device: Target device.
        """
        self.add_self_loops = add_self_loops
        self.device = device or torch.device("cpu")

    def rewire(
        self,
        edge_index: torch.Tensor,
        vgae_embeddings: np.ndarray,
        ratio: float,
        num_nodes: int,
    ) -> torch.Tensor:
        """Remove low-similarity Delaunay edges and add high-similarity VGAE edges.

        Args:
            edge_index: DGlf/Delaunay edge index.
            vgae_embeddings: VGAE embeddings with shape [num_nodes, dim].
            ratio: Percentage of Delaunay edges to replace, in [0, 1].
            num_nodes: Number of nodes.

        Returns:
            Rewired ``edge_index``.
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio precisa estar entre 0 e 1.")
        if vgae_embeddings.shape[0] != num_nodes:
            raise ValueError("Número de embeddings não coincide com num_nodes.")
        if ratio == 0.0:
            return self._finalize(edge_index, num_nodes)

        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edge_index.t().detach().cpu().tolist())
        edges = list(graph.edges())
        k = int(len(edges) * ratio)
        if k <= 0:
            return self._finalize(edge_index, num_nodes)

        sim_matrix = cosine_similarity(vgae_embeddings)
        np.fill_diagonal(sim_matrix, -np.inf)

        edges_sorted = sorted(edges, key=lambda e: sim_matrix[e[0], e[1]])
        graph.remove_edges_from(edges_sorted[:k])

        added = self._add_best_edges(graph, sim_matrix, k, num_nodes)
        if added < k:
            LOGGER.warning("Foram adicionadas %d/%d novas arestas no rewiring.", added, k)

        new_edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        return self._finalize(new_edge_index, num_nodes)

    def _add_best_edges(self, graph: nx.Graph, sim_matrix: np.ndarray, k: int, num_nodes: int) -> int:
        """Add up to k globally most similar non-existing edges."""
        flat = sim_matrix.reshape(-1)
        search_size = min(num_nodes * num_nodes, max(k * 10, (k + graph.number_of_edges()) * 3))
        candidate_ids = np.argpartition(flat, -search_size)[-search_size:]
        candidate_ids = candidate_ids[np.argsort(-flat[candidate_ids])]

        added = 0
        for idx in candidate_ids:
            u, v = divmod(int(idx), num_nodes)
            if u < v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
                added += 1
                if added >= k:
                    break
        return added

    def _finalize(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Convert edge_index to undirected PyG format and optionally add self-loops."""
        edge_index = edge_index.long().contiguous()
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index.to(self.device)
