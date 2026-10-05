"""Hybrid VGAE + commute-time/effective-resistance Delaunay rewiring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from scipy.sparse import coo_matrix, csgraph
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import add_self_loops, to_undirected

from config import RewiringConfig

LOGGER = logging.getLogger("vgae_delaunay_pipeline.hybrid_vgae_ct_rewiring")


@dataclass(frozen=True)
class SpectralResistanceModel:
    """Low-rank spectral model for approximate effective resistance.

    Effective resistance can be written as a squared Euclidean distance in the
    Laplacian pseudoinverse embedding. This class stores the spectral
    coordinates used to evaluate that distance for arbitrary node pairs.
    """

    coordinates: np.ndarray
    eps: float = 1e-12

    def resistance(self, pairs: np.ndarray) -> np.ndarray:
        """Return effective-resistance estimates for node pairs.

        Args:
            pairs: Integer array with shape ``[num_pairs, 2]``.

        Returns:
            One-dimensional array with one resistance value per pair.
        """
        if pairs.size == 0:
            return np.asarray([], dtype=np.float64)
        diffs = self.coordinates[pairs[:, 0]] - self.coordinates[pairs[:, 1]]
        values = np.einsum("ij,ij->i", diffs, diffs)
        return np.maximum(values, self.eps)


class EffectiveResistanceScorer:
    """Build effective-resistance/commute-time scores from a graph.

    For small graphs, the scorer uses a dense eigendecomposition of the graph
    Laplacian. For larger graphs, it uses a truncated sparse eigendecomposition
    with the smallest non-zero Laplacian eigenvalues. This is more practical for
    scientific experimentation than an all-pairs exact pseudoinverse on every
    dataset.
    """

    def __init__(self, config: RewiringConfig) -> None:
        """Initialize the scorer.

        Args:
            config: Rewiring-specific configuration.
        """
        self.config = config

    def fit(self, edge_index: torch.Tensor, num_nodes: int) -> SpectralResistanceModel:
        """Create a spectral resistance model from an undirected edge index.

        Args:
            edge_index: Edge index used as the structural graph.
            num_nodes: Number of nodes.

        Returns:
            A spectral resistance model.
        """
        laplacian = self._build_laplacian(edge_index=edge_index, num_nodes=num_nodes)
        if num_nodes <= self.config.ct_exact_max_nodes:
            LOGGER.info("CT/effective resistance: decomposição exata densa para %d nós.", num_nodes)
            return self._fit_exact(laplacian.toarray())

        LOGGER.info(
            "CT/effective resistance: decomposição espectral aproximada | nós=%d | k=%d.",
            num_nodes,
            self.config.ct_num_eigenvectors,
        )
        return self._fit_approximate(laplacian, num_nodes)

    def _build_laplacian(self, edge_index: torch.Tensor, num_nodes: int):
        """Build an unweighted sparse graph Laplacian."""
        edges = edge_index.detach().cpu().long().numpy()
        rows = edges[0]
        cols = edges[1]
        data = np.ones(rows.shape[0], dtype=np.float64)
        adjacency = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        adjacency = adjacency.maximum(adjacency.T)
        adjacency.setdiag(0.0)
        adjacency.eliminate_zeros()
        return csgraph.laplacian(adjacency, normed=False).astype(np.float64)

    def _fit_exact(self, dense_laplacian: np.ndarray) -> SpectralResistanceModel:
        """Fit exact Laplacian pseudoinverse coordinates."""
        eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
        mask = eigenvalues > self.config.ct_eps
        if not np.any(mask):
            raise ValueError("Não há autovalores não-nulos no Laplaciano; grafo inválido.")
        coordinates = eigenvectors[:, mask] / np.sqrt(eigenvalues[mask])[None, :]
        return SpectralResistanceModel(coordinates=coordinates, eps=self.config.ct_eps)

    def _fit_approximate(self, sparse_laplacian, num_nodes: int) -> SpectralResistanceModel:
        """Fit low-rank Laplacian pseudoinverse coordinates."""
        k = min(max(2, self.config.ct_num_eigenvectors + 1), num_nodes - 1)
        try:
            eigenvalues, eigenvectors = eigsh(sparse_laplacian, k=k, which="SM", tol=1e-3)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("eigsh falhou (%s). Usando decomposição densa como fallback.", exc)
            return self._fit_exact(sparse_laplacian.toarray())

        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        mask = eigenvalues > self.config.ct_eps
        if not np.any(mask):
            raise ValueError("Aproximação CT sem autovalores não-nulos; aumente ct_num_eigenvectors.")
        coordinates = eigenvectors[:, mask] / np.sqrt(eigenvalues[mask])[None, :]
        return SpectralResistanceModel(coordinates=coordinates, eps=self.config.ct_eps)


class HybridVGAECTDelaunayRewirer:
    """Apply hybrid VGAE + CT/effective-resistance Delaunay rewiring.

    The hybrid method preserves the original semantics of the previous pipeline:
    the rewired condition starts from the DGlf/Delaunay graph, and a ratio of
    0% means the pure DGlf/Delaunay graph.

    Removal uses a keep-score:
        alpha * normalized VGAE similarity
        + (1 - alpha) * normalized effective-resistance importance

    Therefore, low-similarity and low-structural-importance Delaunay edges are
    removed first, while likely bridge-like edges are protected.

    Addition uses an add-score:
        alpha * normalized VGAE similarity
        + (1 - alpha) * normalized CT closeness

    where CT closeness is the inverse of effective resistance. This favors new
    edges that are semantically similar and structurally close in the global
    graph geometry.
    """

    def __init__(
        self,
        config: RewiringConfig,
        add_self_loops: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the rewirer.

        Args:
            config: Rewiring-specific configuration.
            add_self_loops: Whether to add self-loops after rewiring.
            device: Target PyTorch device.
        """
        if not 0.0 <= config.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha precisa estar entre 0 e 1.")
        self.config = config
        self.alpha = config.hybrid_alpha
        self.add_self_loops = add_self_loops
        self.device = device or torch.device("cpu")

    def rewire(
        self,
        edge_index: torch.Tensor,
        vgae_embeddings: np.ndarray,
        ratio: float,
        num_nodes: int,
    ) -> torch.Tensor:
        """Remove low hybrid-score Delaunay edges and add high hybrid-score pairs.

        Args:
            edge_index: DGlf/Delaunay edge index.
            vgae_embeddings: VGAE embeddings with shape ``[num_nodes, dim]``.
            ratio: Percentage of Delaunay edges to replace, in ``[0, 1]``.
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

        LOGGER.info(
            "Hybrid rewiring | alpha=%.2f | ratio=%.0f%% | arestas=%d | k=%d",
            self.alpha,
            ratio * 100,
            len(edges),
            k,
        )

        sim_matrix = cosine_similarity(vgae_embeddings)
        np.fill_diagonal(sim_matrix, -np.inf)

        resistance_model = EffectiveResistanceScorer(self.config).fit(edge_index, num_nodes)
        self._remove_low_score_edges(graph, edges, sim_matrix, resistance_model, k)
        added = self._add_high_score_edges(graph, sim_matrix, resistance_model, k, num_nodes)
        if added < k:
            LOGGER.warning("Foram adicionadas %d/%d novas arestas no rewiring híbrido.", added, k)

        new_edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        return self._finalize(new_edge_index, num_nodes)

    def _remove_low_score_edges(
        self,
        graph: nx.Graph,
        edges: list[tuple[int, int]],
        sim_matrix: np.ndarray,
        resistance_model: SpectralResistanceModel,
        k: int,
    ) -> None:
        """Remove edges with the lowest hybrid keep-score."""
        edge_pairs = np.asarray(edges, dtype=np.int64)
        similarities = np.asarray([sim_matrix[u, v] for u, v in edges], dtype=np.float64)
        resistances = resistance_model.resistance(edge_pairs)

        sim_score = self._minmax(similarities)
        resistance_importance = self._minmax(resistances)
        keep_score = self.alpha * sim_score + (1.0 - self.alpha) * resistance_importance

        remove_ids = np.argsort(keep_score)[:k]
        graph.remove_edges_from(edges[idx] for idx in remove_ids)

    def _add_high_score_edges(
        self,
        graph: nx.Graph,
        sim_matrix: np.ndarray,
        resistance_model: SpectralResistanceModel,
        k: int,
        num_nodes: int,
    ) -> int:
        """Add candidate non-edges with the highest hybrid add-score."""
        candidate_pairs = self._candidate_pairs_by_similarity(graph, sim_matrix, k, num_nodes)
        if candidate_pairs.size == 0:
            return 0

        similarities = sim_matrix[candidate_pairs[:, 0], candidate_pairs[:, 1]].astype(np.float64)
        resistances = resistance_model.resistance(candidate_pairs)
        ct_closeness = 1.0 / (resistances + self.config.ct_eps)

        sim_score = self._minmax(similarities)
        ct_score = self._minmax(ct_closeness)
        add_score = self.alpha * sim_score + (1.0 - self.alpha) * ct_score

        ordered = np.argsort(-add_score)
        added = 0
        for idx in ordered:
            u, v = map(int, candidate_pairs[idx])
            if u < v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
                added += 1
                if added >= k:
                    break
        return added

    def _candidate_pairs_by_similarity(
        self,
        graph: nx.Graph,
        sim_matrix: np.ndarray,
        k: int,
        num_nodes: int,
    ) -> np.ndarray:
        """Return a pool of candidate pairs using VGAE similarity as prefilter."""
        flat = sim_matrix.reshape(-1)
        multiplier = max(1, self.config.ct_candidate_multiplier)
        min_factor = max(1, self.config.ct_min_search_factor)
        search_size = min(num_nodes * num_nodes, max(k * multiplier, (k + graph.number_of_edges()) * min_factor))
        candidate_ids = np.argpartition(flat, -search_size)[-search_size:]
        candidate_ids = candidate_ids[np.argsort(-flat[candidate_ids])]

        pairs: list[tuple[int, int]] = []
        for idx in candidate_ids:
            u, v = divmod(int(idx), num_nodes)
            if u < v and not graph.has_edge(u, v):
                pairs.append((u, v))
        return np.asarray(pairs, dtype=np.int64)

    def _finalize(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Convert edge_index to undirected PyG format and optionally add self-loops."""
        edge_index = edge_index.long().contiguous()
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index.to(self.device)

    @staticmethod
    def _minmax(values: Iterable[float] | np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1], returning zeros for constant vectors."""
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return array
        min_value = float(np.nanmin(array))
        max_value = float(np.nanmax(array))
        if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
            return np.zeros_like(array, dtype=np.float64)
        return (array - min_value) / (max_value - min_value)
