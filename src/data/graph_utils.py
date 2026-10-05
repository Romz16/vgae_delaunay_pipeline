
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, from_networkx, remove_self_loops, to_undirected


def ensure_edge_index(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        raise ValueError("edge_index vazio ou ausente.")
    edge_index = edge_index.long()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index.contiguous().to(device)


def ensure_features(data: Data, device: torch.device) -> Data:
    if getattr(data, "x", None) is not None:
        data.x = data.x.float().to(device)
        return data

    deg = degree(data.edge_index[0].detach().cpu(), num_nodes=data.num_nodes).numpy()
    features = np.stack(
        [
            deg,
            np.log1p(deg),
            (deg == 0).astype(float),
        ],
        axis=1,
    )
    data.x = torch.as_tensor(features, dtype=torch.float32, device=device)
    return data


def normalize_pyg_data(data: Data, device: torch.device) -> Data:
    if data.num_nodes is None:
        raise ValueError("data.num_nodes não pôde ser inferido.")
    data.edge_index = ensure_edge_index(data.edge_index, int(data.num_nodes), device)
    data = ensure_features(data, device)
    if getattr(data, "y", None) is None:
        raise ValueError("O grafo precisa possuir labels em `data.y`.")
    data.y = data.y.long().to(device)
    return data.to(device)


def networkx_to_pyg(graph, label_attr: str = "y", feature_attr: str = "x") -> Data:
    data = from_networkx(graph)
    if hasattr(data, feature_attr):
        data.x = getattr(data, feature_attr)
    if hasattr(data, label_attr):
        data.y = getattr(data, label_attr)
    return data
