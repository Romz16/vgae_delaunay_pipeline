
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from config import SplitConfig


def apply_node_split(data: Data, seed: int, config: SplitConfig, device: torch.device) -> Data:
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


def apply_nested_node_split(
    data: Data,
    seed: int,
    device: torch.device,
    *,
    test_size: float = 0.20,
    inner_val_size: float = 0.10,
    topology_val_size: float = 0.10,
    stratify: bool = True,
) -> Data:
    fractions = (test_size, inner_val_size, topology_val_size)
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("All nested split fractions must be between 0 and 1.")
    train_size = 1.0 - sum(fractions)
    if train_size <= 0.0:
        raise ValueError("Nested split fractions must leave a non-empty training set.")
    if not hasattr(data, "y") or data.y is None:
        raise ValueError("Node labels `data.y` are required for node classification.")

    num_nodes = int(data.num_nodes)
    indices = np.arange(num_nodes)
    labels = data.y.detach().cpu().numpy()

    def split(values: np.ndarray, held_out_count: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        labels_subset = labels[values] if stratify else None
        try:
            return train_test_split(
                values,
                test_size=held_out_count,
                random_state=random_state,
                stratify=labels_subset,
            )
        except ValueError:
            return train_test_split(values, test_size=held_out_count, random_state=random_state)

    test_count = max(1, int(round(num_nodes * test_size)))
    topology_count = max(1, int(round(num_nodes * topology_val_size)))
    inner_count = max(1, int(round(num_nodes * inner_val_size)))
    if test_count + topology_count + inner_count >= num_nodes:
        raise ValueError("Nested split sizes leave no observations for training.")
    remaining_idx, test_idx = split(indices, test_count, seed)
    remaining_idx, topology_val_idx = split(remaining_idx, topology_count, seed + 1)
    train_idx, inner_val_idx = split(remaining_idx, inner_count, seed + 2)

    def make_mask(selected: np.ndarray) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[torch.as_tensor(selected, dtype=torch.long, device=device)] = True
        return mask

    data.train_mask = make_mask(train_idx)
    data.inner_val_mask = make_mask(inner_val_idx)
    data.topology_val_mask = make_mask(topology_val_idx)
    data.test_mask = make_mask(test_idx)
    data.val_mask = data.inner_val_mask.clone()
    return data
