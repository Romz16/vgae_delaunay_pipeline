"""Modern PyTorch compatibility pilot for the IDGL graph-learning baseline.

This is a documented compatibility port of IDGL's core ingredients (a
multi-perspective weighted-cosine graph learner, skip connection to the observed
adjacency, and a GCN trained jointly with the learned graph).  It deliberately
uses the project's stratified 60/20/20 splits and chooses checkpoints by
validation macro-F1 only.  It is a pilot, not a claim of bit-for-bit reproduction
of the 2020 PyTorch-0.4 implementation.
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


PROJECT = Path("/media/work/romulorocha/vgae_delaunay_hybrid_pipeline")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from config import SplitConfig
from src.data.loaders import GraphLoader
from src.data.splits import apply_node_split


class WeightedCosineGraphLearner(nn.Module):
    """The official IDGL weighted-cosine learner, expressed in current PyTorch."""

    def __init__(self, features: int, perspectives: int = 4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(perspectives, features))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = F.normalize(x.unsqueeze(0) * self.weights.unsqueeze(1), p=2, dim=-1)
        similarity = torch.matmul(scaled, scaled.transpose(1, 2)).mean(dim=0)
        similarity = similarity.clamp_min(0.0)
        return similarity / similarity.sum(dim=1, keepdim=True).clamp_min(1e-12)


class DenseGCN(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.first = nn.Linear(in_features, hidden, bias=False)
        self.last = nn.Linear(hidden, classes, bias=False)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, adj: torch.Tensor, dropout: float | None = None) -> torch.Tensor:
        hidden = F.relu(adj @ self.first(x))
        return F.dropout(hidden, p=self.dropout if dropout is None else dropout, training=self.training)

    def classify(self, hidden: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return adj @ self.last(hidden)


class IDGLCompat(nn.Module):
    def __init__(self, in_features: int, hidden: int, classes: int, dropout: float, skip: float, max_iter: int) -> None:
        super().__init__()
        self.learner = WeightedCosineGraphLearner(in_features)
        self.learner2 = WeightedCosineGraphLearner(hidden)
        self.gcn = DenseGCN(in_features, hidden, classes, dropout)
        self.skip = skip
        self.max_iter = max_iter

    def forward(self, x: torch.Tensor, observed_norm: torch.Tensor) -> torch.Tensor:
        # This follows IDGL's full-graph loop: learn from features first, then
        # repeatedly relearn the graph from the current node representation.
        first_learned = self.learner(x)
        adj = self.skip * observed_norm + (1.0 - self.skip) * first_learned
        first_adj = adj
        hidden = self.gcn.encode(x, adj)
        for _ in range(self.max_iter):
            learned = self.learner2(hidden)
            refined = self.skip * observed_norm + (1.0 - self.skip) * learned
            # Official IDGL uses update_adj_ratio=0.1 to stabilize refinement.
            # Keep the first learned adjacency as the anchor.  This matches
            # IDGL's update_adj_ratio rule (rather than accumulating drift
            # from the preceding refinement step).
            adj = 0.1 * refined + 0.9 * first_adj
            hidden = self.gcn.encode(x, adj)
        return self.gcn.classify(hidden, adj)


def normalized_observed_adj(edge_index: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    adj = torch.zeros((n, n), device=device)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj.fill_diagonal_(1.0)
    degree = adj.sum(dim=1).clamp_min(1.0)
    return adj * degree.rsqrt().unsqueeze(1) * degree.rsqrt().unsqueeze(0)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, observed: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    model.eval()
    pred = model(x, observed).argmax(dim=1)[mask].detach().cpu().numpy()
    gold = y[mask].detach().cpu().numpy()
    return float(accuracy_score(gold, pred)), float(f1_score(gold, pred, average="macro", zero_division=0))


def run_one(dataset: str, seed: int, args: argparse.Namespace, device: torch.device) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    loader = GraphLoader(root=PROJECT / "data", device=device)
    data = loader.load_from_name(dataset)
    data = apply_node_split(data, seed, SplitConfig(), device)
    x = data.x.float().to(device)
    y = data.y.long().to(device)
    observed = normalized_observed_adj(data.edge_index.to(device), data.num_nodes, device)
    model = IDGLCompat(x.size(1), args.hidden, int(y.max().item()) + 1, args.dropout, args.skip, args.max_iter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_val, best_epoch, stale = None, -float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, observed)
        loss = F.cross_entropy(logits[data.train_mask], y[data.train_mask])
        loss.backward()
        optimizer.step()
        _, val_f1 = evaluate(model, x, observed, y, data.val_mask)
        if val_f1 > best_val + 1e-12:
            best_val, best_epoch, stale = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_acc, val_f1 = evaluate(model, x, observed, y, data.val_mask)
    test_acc, test_f1 = evaluate(model, x, observed, y, data.test_mask)
    return {"dataset": dataset, "backbone": "idgl_compat_gcn", "seed": seed,
            "val_acc": val_acc, "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1,
            "best_epoch": best_epoch, "hidden": args.hidden, "lr": args.lr,
            "skip": args.skip, "dropout": args.dropout}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=12345)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--skip", type=float, default=0.8)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset.lower() not in {"cora", "cornell", "texas"}:
        raise ValueError("Pilot supports Cora, Cornell, and Texas only.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_one(args.dataset, args.seed_start + i, args, device) for i in range(args.runs)]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for metric in ("test_acc", "test_f1", "val_f1"):
        values = np.array([float(row[metric]) for row in rows])
        print(f"{metric}: mean={values.mean():.6f} std={values.std(ddof=1):.6f}")
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
