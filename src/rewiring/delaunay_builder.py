"""DGlf construction: auxiliary GCN, UMAP projection and Delaunay graph."""

from __future__ import annotations

import copy
import logging
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import umap
from scipy.spatial import Delaunay, QhullError
from sklearn.manifold import trustworthiness
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from config import PipelineConfig
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed

LOGGER = logging.getLogger("vgae_delaunay_pipeline.delaunay_builder")


class AuxiliaryGCN(torch.nn.Module):
    """Small supervised GCN used to extract learned features for DGlf."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        """Initialize the auxiliary GCN."""
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return first-layer learned features."""
        return F.relu(self.conv1(x, edge_index))


class DelaunayGraphBuilder:
    """Build a Delaunay graph from GCN learned features projected with UMAP."""

    def __init__(self, config: PipelineConfig, device: torch.device) -> None:
        """Initialize the builder.

        Args:
            config: Global pipeline configuration.
            device: Target device.
        """
        self.config = config
        self.device = device

    def build(self, data: Data) -> torch.Tensor:
        """Build the DGlf Delaunay graph.

        Args:
            data: Graph with train/validation masks already attached.

        Returns:
            Delaunay graph as PyG ``edge_index``.
        """
        edge_index, _, _ = self.build_with_projection(data)
        return edge_index

    def build_with_projection(self, data: Data) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Build DGlf and return edge_index, learned features and UMAP points."""
        LOGGER.info("Treinando GCN auxiliar para extrair learned features DGlf...")
        learned_features = self._extract_learned_features(data)
        LOGGER.info("Projetando learned features com UMAP...")
        points_2d = self._project_umap(learned_features)
        LOGGER.info("Construindo grafo Delaunay...")
        return self._build_delaunay_edges(points_2d, data.num_nodes), learned_features, points_2d

    def fit_auxiliary_gcn(
        self,
        data: Data,
        *,
        seed: int | None = None,
        train_mask: torch.Tensor | None = None,
        validation_mask: torch.Tensor | None = None,
        selection_metric: str = "f1",
    ) -> tuple[AuxiliaryGCN, dict[str, float]]:
        """Fit the feature encoder and expose its classifier-only baseline."""
        if seed is not None:
            set_seed(seed)
        train_mask = data.train_mask if train_mask is None else train_mask
        validation_mask = data.val_mask if validation_mask is None else validation_mask
        selection_metric = selection_metric.lower()
        if selection_metric not in {"acc", "f1"}:
            raise ValueError("selection_metric must be 'acc' or 'f1'.")

        model = AuxiliaryGCN(
            in_channels=int(data.num_features),
            hidden_channels=self.config.delaunay.pretrain_hidden_channels,
            out_channels=int(data.y.max().item() + 1),
        ).to(self.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.delaunay.pretrain_lr,
            weight_decay=self.config.delaunay.pretrain_weight_decay,
        )
        best_value = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        started = time.perf_counter()

        for epoch in range(1, self.config.delaunay.pretrain_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = model(data.x, data.edge_index).argmax(dim=1)
            metrics = classification_metrics(
                data.y[validation_mask].detach().cpu().numpy(),
                predictions[validation_mask].detach().cpu().numpy(),
            )
            if metrics[selection_metric] > best_value:
                best_value = metrics[selection_metric]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("Auxiliary GCN did not produce a valid checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions = model(data.x, data.edge_index).argmax(dim=1)

        result: dict[str, float] = {
            "best_epoch": float(best_epoch),
            "selection_value": float(best_value),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        report_masks = {
            "inner_val": validation_mask,
            "topology_val": getattr(data, "topology_val_mask", None),
            "test": getattr(data, "test_mask", None),
        }
        for name, mask in report_masks.items():
            if mask is None:
                continue
            metrics = classification_metrics(
                data.y[mask].detach().cpu().numpy(),
                predictions[mask].detach().cpu().numpy(),
            )
            result[f"{name}_acc"] = metrics["acc"]
            result[f"{name}_f1"] = metrics["f1"]
        return model, result

    def build_from_auxiliary_model(
        self, data: Data, model: AuxiliaryGCN
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Build Delaunay from a previously selected auxiliary checkpoint."""
        model.eval()
        with torch.no_grad():
            features = model.get_embeddings(data.x, data.edge_index)
        learned_features = features.detach().cpu().numpy().astype(np.float32, copy=False)
        points_2d = self._project_umap(learned_features)
        edge_index = self._build_delaunay_edges(points_2d, data.num_nodes)
        return edge_index, learned_features, points_2d

    def build_from_raw_features(self, data: Data) -> torch.Tensor:
        """Build a raw-feature UMAP -> Delaunay baseline graph."""
        edge_index, _, _ = self.build_from_raw_features_with_projection(data)
        return edge_index

    def build_from_raw_features_with_projection(self, data: Data) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Build the raw-feature baseline and return its features and UMAP points."""
        features = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
        points_2d = self._project_umap(features)
        edge_index = self._build_delaunay_edges(points_2d, data.num_nodes)
        return edge_index, features, points_2d

    def umap_trustworthiness(self, original_features: np.ndarray, points_2d: np.ndarray) -> float:
        """Compute UMAP trustworthiness between high-dimensional features and 2D projection."""
        n_samples = int(original_features.shape[0])
        max_samples = int(self.config.delaunay.trustworthiness_max_samples)
        if n_samples > max_samples:
            rng = np.random.default_rng(self.config.delaunay.umap_seed)
            selected = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
            original_features = original_features[selected]
            points_2d = points_2d[selected]
            n_samples = max_samples
        max_neighbors = max(1, (n_samples // 2) - 1)
        n_neighbors = min(self.config.delaunay.umap_neighbors, max_neighbors)
        return float(trustworthiness(original_features, points_2d, n_neighbors=n_neighbors))

    def _extract_learned_features(self, data: Data) -> np.ndarray:
        """Train the auxiliary GCN and extract hidden representations."""
        model, metrics = self.fit_auxiliary_gcn(data, selection_metric="acc")
        LOGGER.info("GCN auxiliar concluída | Val Acc: %.4f", metrics["inner_val_acc"])
        model.eval()
        with torch.no_grad():
            features = model.get_embeddings(data.x, data.edge_index)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def _project_umap(self, features: np.ndarray) -> np.ndarray:
        """Project features to two dimensions using UMAP."""
        reducer = umap.UMAP(
            n_components=self.config.delaunay.umap_components,
            n_neighbors=self.config.delaunay.umap_neighbors,
            min_dist=self.config.delaunay.umap_min_dist,
            random_state=self.config.delaunay.umap_seed,
        )
        return reducer.fit_transform(features)

    def _build_delaunay_edges(self, points_2d: np.ndarray, num_nodes: int) -> torch.Tensor:
        """Build a Delaunay edge_index from 2D points."""
        if points_2d.shape[1] != 2:
            raise ValueError("Delaunay exige projeção 2D.")
        try:
            triangulation = Delaunay(points_2d)
        except QhullError as exc:
            raise RuntimeError("Falha ao construir Delaunay. Verifique a projeção UMAP.") from exc

        edges: set[tuple[int, int]] = set()
        for simplex in triangulation.simplices:
            for i, j in self._simplex_pairs(simplex):
                u, v = sorted((int(i), int(j)))
                if u != v:
                    edges.add((u, v))

        if not edges:
            raise RuntimeError("Delaunay não gerou arestas.")

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        LOGGER.info("DGlf/Delaunay gerado | Arestas direcionadas PyG: %d", edge_index.size(1))
        return edge_index.to(self.device)

    @staticmethod
    def _simplex_pairs(simplex: Iterable[int]) -> list[tuple[int, int]]:
        """Return undirected edge pairs for a triangle simplex."""
        s = list(simplex)
        return [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]
