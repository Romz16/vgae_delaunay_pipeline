
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_geometric.data import Data

from config import PipelineConfig
from src.data.splits import apply_node_split
from src.models.gnn import GNNParams, build_gnn
from src.utils.metrics import classification_metrics, metric_summary
from src.utils.seed import set_seed


class GNNTrainer:

    def __init__(self, config: PipelineConfig, device: torch.device) -> None:

        self.config = config
        self.device = device

    def train_single_run(
        self,
        data: Data,
        backbone: str,
        edge_index: torch.Tensor,
        params: GNNParams,
        seed: int,
        epochs: int,
        track_test: bool = True,
    ) -> dict[str, float]:

        set_seed(seed)
        data = apply_node_split(data, seed=seed, config=self.config.split, device=self.device)
        edge_index = edge_index.to(self.device)
        in_channels = int(data.num_features)
        out_channels = int(data.y.max().item() + 1)

        model = build_gnn(backbone, in_channels, out_channels, params, self.device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        scheduler = self._build_scheduler(optimizer, params, epochs)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_acc = -1.0
        best_val_f1 = -1.0
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_loss = 0.0

        for _ in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.x, edge_index)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                logits = model(data.x, edge_index)
                preds = logits.argmax(dim=1)
                val_metrics = classification_metrics(
                    data.y[data.val_mask].detach().cpu().numpy(),
                    preds[data.val_mask].detach().cpu().numpy(),
                )
                if val_metrics["acc"] > best_val_acc:
                    best_val_acc = val_metrics["acc"]
                    best_val_f1 = val_metrics["f1"]
                    best_loss = float(loss.detach().cpu().item())
                    best_state = copy.deepcopy(model.state_dict())
                    if track_test:
                        test_metrics = classification_metrics(
                            data.y[data.test_mask].detach().cpu().numpy(),
                            preds[data.test_mask].detach().cpu().numpy(),
                        )
                        best_test_acc = test_metrics["acc"]
                        best_test_f1 = test_metrics["f1"]

        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "val_acc": best_val_acc,
            "val_f1": best_val_f1,
            "test_acc": best_test_acc,
            "test_f1": best_test_f1,
            "loss": best_loss,
        }

    def evaluate_repeated_runs(
        self,
        data: Data,
        backbone: str,
        edge_index: torch.Tensor,
        params: GNNParams,
        num_runs: int,
        epochs: int,
    ) -> dict[str, object]:
        
        accs: list[float] = []
        f1s: list[float] = []
        losses: list[float] = []
        start = time.time()

        for run_idx in range(num_runs):
            seed = self.config.run_seed_start + run_idx
            metrics = self.train_single_run(
                data=data,
                backbone=backbone,
                edge_index=edge_index,
                params=params,
                seed=seed,
                epochs=epochs,
                track_test=True,
            )
            accs.append(float(metrics["test_acc"]))
            f1s.append(float(metrics["test_f1"]))
            losses.append(float(metrics["loss"]))

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "accs": accs,
            "f1s": f1s,
            "losses": losses,
            "acc_summary": metric_summary(accs),
            "f1_summary": metric_summary(f1s),
            "loss_summary": metric_summary(losses),
            "elapsed_seconds": time.time() - start,
        }

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
        params: GNNParams,
        epochs: int,
    ):

        scheduler = params.scheduler.lower()
        if scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler == "step":
            return StepLR(
                optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.step_gamma,
            )
        if scheduler == "none":
            return None
        raise ValueError(f"Scheduler desconhecido: {params.scheduler}")
