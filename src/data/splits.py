"""Train/validation/test masks for node classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from config import SplitConfig


def apply_node_split(data: Data, seed: int, config: SplitConfig, device: torch.device) -> Data:
    """Attach stratified train/validation/test masks to a PyG graph.

    Args:
        data: PyTorch Geometric graph with ``y`` labels.
        seed: Random seed for the split.
        config: Split configuration.
        device: Target device for masks.

    Returns:
        The same graph object with ``train_mask``, ``val_mask`` and
        ``test_mask`` attributes.
    """
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Node labels `data.y` are required for node classification.")

    num_nodes = data.num_nodes
    idx = np.arange(num_nodes)
    y = data.y.detach().cpu().numpy()
    stratify = y if config.stratify else None

    try:
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=config.test_size,
            random_state=seed,
            stratify=stratify,
        )
        train_stratify = y[train_val_idx] if config.stratify else None
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=config.val_size_from_train_val,
            random_state=seed,
            stratify=train_stratify,
        )
    except ValueError:
        train_val_idx, test_idx = train_test_split(idx, test_size=config.test_size, random_state=seed)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=config.val_size_from_train_val,
            random_state=seed,
        )

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    data.train_mask[torch.as_tensor(train_idx, device=device)] = True
    data.val_mask[torch.as_tensor(val_idx, device=device)] = True
    data.test_mask[torch.as_tensor(test_idx, device=device)] = True
    return data
