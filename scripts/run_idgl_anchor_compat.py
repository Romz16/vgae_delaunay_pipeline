"""Scalable IDGL-anchor compatibility baseline for the project protocol.

This uses IDGL's node-to-anchor weighted-cosine graph learner and its
node-anchor-node message passing.  It avoids an N x N learned adjacency, so it
can be run on every benchmark in this repository.  Splits and early stopping
are exactly the project stratified 60/20/20 protocol; the output keeps val/test
per seed so an original-vs-IDGL fallback can be selected by validation only.

It is a modern-PyTorch compatibility implementation, not a byte-identical run
of the legacy IDGL release (PyTorch 0.4).  Results must carry that label.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import GCNConv

PROJECT = Path("/media/work/romulorocha/vgae_delaunay_hybrid_pipeline")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from config import SplitConfig
from src.data.loaders import GraphLoader
from src.data.splits import apply_node_split


class AnchorWeightedCosine(nn.Module):
    """IDGL AnchorGraphLearner's weighted-cosine score, with top-k sparsity."""

    def __init__(self, features: int, perspectives: int, topk: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(perspectives, features))
        nn.init.xavier_uniform_(self.weight)
        self.topk = topk

    def forward(self, nodes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        node_repr = F.normalize(nodes.unsqueeze(0) * self.weight.unsqueeze(1), p=2, dim=-1)
        anchor_repr = F.normalize(anchors.unsqueeze(0) * self.weight.unsqueeze(1), p=2, dim=-1)
        scores = torch.matmul(node_repr, anchor_repr.transpose(1, 2)).mean(dim=0).clamp_min(0.0)
        k = min(self.topk, scores.size(1))
        values, indices = scores.topk(k, dim=1)
        sparse_scores = torch.zeros_like(scores).scatter_(1, indices, values)
        return sparse_scores / sparse_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)


def anchor_propagate(x: torch.Tensor, node_anchor: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
    """The node-anchor-node normalization used by IDGL's AnchorGCN layer."""
    support = layer(x)
    node_to_anchor = node_anchor / node_anchor.sum(dim=0, keepdim=True).clamp_min(1e-12)
    anchor_to_node = node_anchor / node_anchor.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return anchor_to_node @ (node_to_anchor.transpose(0, 1) @ support)


class IDGLAnchorCompat(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float,
                 skip: float, perspectives: int, topk: int, max_iter: int) -> None:
        super().__init__()
        self.learner1 = AnchorWeightedCosine(in_features, perspectives, topk)
        self.learner2 = AnchorWeightedCosine(hidden, perspectives, topk)
        self.anchor1 = nn.Linear(in_features, hidden, bias=False)
        self.anchor_hidden = nn.Linear(hidden, hidden, bias=False)
        self.anchor_out = nn.Linear(hidden, classes, bias=False)
        self.observed1 = GCNConv(in_features, hidden, add_self_loops=True, normalize=True)
        self.observed_hidden = GCNConv(hidden, hidden, add_self_loops=True, normalize=True)
        self.observed_out = GCNConv(hidden, classes, add_self_loops=True, normalize=True)
        self.dropout, self.skip, self.max_iter = dropout, skip, max_iter

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, anchor_index: torch.Tensor) -> torch.Tensor:
        weights = self.learner1(x, x[anchor_index])
        hidden = (1.0 - self.skip) * anchor_propagate(x, weights, self.anchor1) + self.skip * self.observed1(x, edge_index)
        hidden = F.dropout(F.relu(hidden), p=self.dropout, training=self.training)
        first_weights = weights
        for _ in range(self.max_iter):
            refined = self.learner2(hidden, hidden[anchor_index])
            # The official update_adj_ratio anchors refinement to first graph.
            weights = 0.1 * refined + 0.9 * first_weights
            hidden = (1.0 - self.skip) * anchor_propagate(hidden, weights, self.anchor_hidden) + self.skip * self.observed_hidden(hidden, edge_index)
            hidden = F.dropout(F.relu(hidden), p=self.dropout, training=self.training)
        return (1.0 - self.skip) * anchor_propagate(hidden, weights, self.anchor_out) + self.skip * self.observed_out(hidden, edge_index)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, edges: torch.Tensor, anchors: torch.Tensor,
             y: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    model.eval()
    pred = model(x, edges, anchors).argmax(dim=1)[mask].cpu().numpy()
    gold = y[mask].cpu().numpy()
    return float(accuracy_score(gold, pred)), float(f1_score(gold, pred, average="macro", zero_division=0))


def run_one(dataset: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    data = GraphLoader(root=PROJECT / "data", device=device).load_from_name(dataset)
    data = apply_node_split(data, seed, SplitConfig(), device)
    x, y, edges = data.x.float().to(device), data.y.long().to(device), data.edge_index.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    anchors = torch.randperm(data.num_nodes, generator=generator, device=device)[:min(args.anchors, data.num_nodes)]
    model = IDGLAnchorCompat(x.size(1), args.hidden, int(y.max()) + 1, args.dropout,
                             args.skip, args.perspectives, args.topk, args.max_iter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_val, best_epoch, stale = None, -float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, edges, anchors)[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()
        _, val_f1 = evaluate(model, x, edges, anchors, y, data.val_mask)
        if val_f1 > best_val + 1e-12:
            best_val, best_epoch, stale = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    val_acc, val_f1 = evaluate(model, x, edges, anchors, y, data.val_mask)
    test_acc, test_f1 = evaluate(model, x, edges, anchors, y, data.test_mask)
    return {"dataset": dataset, "backbone": "idgl_anchor_compat_gcn", "seed": seed,
            "val_acc": val_acc, "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1,
            "best_epoch": best_epoch, "anchors": len(anchors), "topk": args.topk,
            "max_iter": args.max_iter, "skip": args.skip}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--skip", type=float, default=0.8)
    parser.add_argument("--perspectives", type=int, default=4)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    supported = {"cora", "coauthor-cs", "cornell", "airports-europe", "minesweeper", "actor", "airports-usa", "airports-brazil", "roman-empire", "amazon-photo", "pubmed", "texas"}
    if args.dataset.lower() not in supported:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [run_one(args.dataset, args.seed_start + offset, args, device) for offset in range(args.runs)]
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    for metric in ("test_acc", "test_f1", "val_f1"):
        values = np.asarray([float(row[metric]) for row in rows])
        print(f"{metric}: mean={values.mean():.6f} std={values.std(ddof=1):.6f}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
