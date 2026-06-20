from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE


@dataclass(frozen=True)
class VGAEParams:

    hidden_channels: int
    latent_channels: int
    lr: float
    weight_decay: float
    dropout: float


class GCNVGAEEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        dropout: float,
    ) -> None:

        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_logstd = GCNConv(hidden_channels, latent_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log standard deviation."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def build_vgae(in_channels: int, params: VGAEParams, device: torch.device) -> VGAE:

    encoder = GCNVGAEEncoder(
        in_channels=in_channels,
        hidden_channels=params.hidden_channels,
        latent_channels=params.latent_channels,
        dropout=params.dropout,
    )
    return VGAE(encoder).to(device)
