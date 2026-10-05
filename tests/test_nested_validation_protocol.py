from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.splits import apply_nested_node_split


def test_nested_split_is_disjoint_complete_and_reproducible() -> None:
    labels = torch.tensor(np.repeat(np.arange(4), 50), dtype=torch.long)
    base = Data(x=torch.ones((200, 3)), y=labels, num_nodes=200)
    first = apply_nested_node_split(base.clone(), 17, torch.device("cpu"))
    second = apply_nested_node_split(base.clone(), 17, torch.device("cpu"))

    masks = [first.train_mask, first.inner_val_mask, first.topology_val_mask, first.test_mask]
    assert [int(mask.sum()) for mask in masks] == [120, 20, 20, 40]
    assert torch.stack(masks).sum(dim=0).eq(1).all()
    assert torch.equal(first.val_mask, first.inner_val_mask)
    for name in ("train_mask", "inner_val_mask", "topology_val_mask", "test_mask"):
        assert torch.equal(getattr(first, name), getattr(second, name))


def test_nested_split_changes_with_seed() -> None:
    labels = torch.tensor(np.repeat(np.arange(2), 50), dtype=torch.long)
    base = Data(x=torch.ones((100, 2)), y=labels, num_nodes=100)
    first = apply_nested_node_split(base.clone(), 1, torch.device("cpu"))
    second = apply_nested_node_split(base.clone(), 2, torch.device("cpu"))
    assert not torch.equal(first.test_mask, second.test_mask)
