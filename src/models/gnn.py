"""Flexible node-classification GNN architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


@dataclass(frozen=True)
class GNNParams:
    """Hyperparameters for a GNN backbone."""

    hidden_channels: int
    n_layers: int
    lr: float
    weight_decay: float
    dropout: float
    activation: str
    batch_norm: bool
    scheduler: str = "none"
    heads: int | None = None


def activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:

    key = name.lower()
    if key == "relu":
        return F.relu
    if key == "leaky_relu":
        return F.leaky_relu
    if key == "elu":
        return F.elu
    if key == "gelu":
        return F.gelu
    raise ValueError(f"Ativação desconhecida: {name}")


class FlexibleGNN(torch.nn.Module):

    def __init__(
        self,
        kind: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        dropout: float,
        activation: str,
        batch_norm: bool,
        heads: int | None = None,
        residual: bool = False,
    ) -> None:
        """Initialize a flexible GNN.

        Args:
            kind: ``gcn``, ``sage`` or ``gat``.
            in_channels: Number of input node features.
            hidden_channels: Hidden dimensionality.
            out_channels: Number of classes.
            n_layers: Number of graph convolution layers.
            dropout: Dropout probability.
            activation: Activation function name.
            batch_norm: Whether to use BatchNorm after hidden convolutions.
            heads: Attention heads for GAT hidden layers.
            residual: Whether to add residual connections when shapes match.
        """
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers precisa ser >= 1.")

        self.kind = kind.lower()
        self.dropout = dropout
        self.act = activation_fn(activation)
        self.residual = residual
        self.heads = heads or 1
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if n_layers == 1:
            self.convs.append(self._make_conv(in_channels, out_channels, is_last=True))
            return

        self.convs.append(self._make_conv(in_channels, hidden_channels, is_last=False))
        self.norms.append(
            self._make_norm(self._hidden_dim(hidden_channels), batch_norm)
        )

        for _ in range(n_layers - 2):
            self.convs.append(
                self._make_conv(
                    self._hidden_dim(hidden_channels), hidden_channels, is_last=False
                )
            )
            self.norms.append(
                self._make_norm(self._hidden_dim(hidden_channels), batch_norm)
            )

        self.convs.append(
            self._make_conv(
                self._hidden_dim(hidden_channels), out_channels, is_last=True
            )
        )

    def _hidden_dim(self, hidden_channels: int) -> int:
        """Return hidden output dimension for the current backbone."""
        return hidden_channels * self.heads if self.kind == "gat" else hidden_channels

    @staticmethod
    def _make_norm(channels: int, batch_norm: bool) -> torch.nn.Module:
        """Create a normalization module."""
        return torch.nn.BatchNorm1d(channels) if batch_norm else torch.nn.Identity()

    def _make_conv(
        self, in_channels: int, out_channels: int, is_last: bool
    ) -> torch.nn.Module:
        """Create one graph convolution layer."""
        if self.kind == "gcn":
            return GCNConv(in_channels, out_channels)
        if self.kind == "sage":
            return SAGEConv(in_channels, out_channels)
        if self.kind == "gat":
            heads = 1 if is_last else self.heads
            return GATConv(in_channels, out_channels, heads=heads)
        raise ValueError(f"Backbone desconhecido: {self.kind}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Node feature matrix.
            edge_index: Graph connectivity.

        Returns:
            Class logits for each node.
        """
        for layer_idx, conv in enumerate(self.convs[:-1]):
            previous = x
            x = conv(x, edge_index)
            x = self.norms[layer_idx](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and previous.shape == x.shape:
                x = x + previous
        return self.convs[-1](x, edge_index)


def build_gnn(
    backbone: str,
    in_channels: int,
    out_channels: int,
    params: GNNParams,
    device: torch.device,
) -> FlexibleGNN:

    normalized = backbone.lower()
    residual = normalized == "gcn_residual"
    kind = "gcn" if residual else normalized
    return FlexibleGNN(
        kind=kind,
        in_channels=in_channels,
        hidden_channels=params.hidden_channels,
        out_channels=out_channels,
        n_layers=params.n_layers,
        dropout=params.dropout,
        activation=params.activation,
        batch_norm=params.batch_norm,
        heads=params.heads,
        residual=residual,
    ).to(device)
