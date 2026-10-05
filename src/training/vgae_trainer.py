
from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import VGAE

from src.models.vgae import VGAEParams, build_vgae
from src.utils.seed import set_seed


class VGAETrainer:

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def train(
        self,
        data: Data,
        params: VGAEParams,
        epochs: int,
        seed: int,
        edge_index: torch.Tensor | None = None,
    ) -> VGAE:
        set_seed(seed)
        graph_edges = (edge_index if edge_index is not None else data.edge_index).to(self.device)
        model = build_vgae(int(data.num_features), params, self.device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        for _ in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(data.x, graph_edges)
            loss = model.recon_loss(z, graph_edges)
            loss = loss + (1.0 / data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()

        return model

    @torch.no_grad()
    def extract_embeddings(self, model: VGAE, data: Data, edge_index: torch.Tensor | None = None) -> np.ndarray:
        model.eval()
        graph_edges = (edge_index if edge_index is not None else data.edge_index).to(self.device)
        z = model.encode(data.x, graph_edges)
        return z.detach().cpu().numpy().astype(np.float32, copy=False)
