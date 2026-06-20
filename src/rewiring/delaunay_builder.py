from __future__ import annotations

import copy
import logging
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import umap
from scipy.spatial import Delaunay, QhullError
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from config import PipelineConfig

LOGGER = logging.getLogger("vgae_delaunay_pipeline.delaunay_builder")


class AuxiliaryGCN(torch.nn.Module):

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int
    ) -> None:
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

    def __init__(self, config: PipelineConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

    def build(self, data: Data) -> torch.Tensor:

        LOGGER.info("Treinando GCN auxiliar para extrair learned features DGlf...")
        learned_features = self._extract_learned_features(data)
        LOGGER.info("Projetando learned features com UMAP...")
        points_2d = self._project_umap(learned_features)
        LOGGER.info("Construindo grafo Delaunay...")
        return self._build_delaunay_edges(points_2d, data.num_nodes)

    def _extract_learned_features(self, data: Data) -> np.ndarray:

        in_channels = int(data.num_features)
        out_channels = int(data.y.max().item() + 1)
        model = AuxiliaryGCN(
            in_channels=in_channels,
            hidden_channels=self.config.delaunay.pretrain_hidden_channels,
            out_channels=out_channels,
        ).to(self.device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.delaunay.pretrain_lr,
            weight_decay=self.config.delaunay.pretrain_weight_decay,
        )
        best_val_acc = -1.0
        best_state: dict[str, torch.Tensor] | None = None

        for _ in range(1, self.config.delaunay.pretrain_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(data.x, data.edge_index).argmax(dim=1)
                val_acc = (
                    (preds[data.val_mask] == data.y[data.val_mask])
                    .float()
                    .mean()
                    .item()
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())

        LOGGER.info("GCN auxiliar concluída | Val Acc: %.4f", best_val_acc)
        if best_state is not None:
            model.load_state_dict(best_state)
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

    def _build_delaunay_edges(
        self, points_2d: np.ndarray, num_nodes: int
    ) -> torch.Tensor:

        if points_2d.shape[1] != 2:
            raise ValueError("Delaunay exige projeção 2D.")
        try:
            triangulation = Delaunay(points_2d)
        except QhullError as exc:
            raise RuntimeError(
                "Falha ao construir Delaunay. Verifique a projeção UMAP."
            ) from exc

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
        LOGGER.info(
            "DGlf/Delaunay gerado | Arestas direcionadas PyG: %d", edge_index.size(1)
        )
        return edge_index.to(self.device)

    @staticmethod
    def _simplex_pairs(simplex: Iterable[int]) -> list[tuple[int, int]]:

        s = list(simplex)
        return [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]
