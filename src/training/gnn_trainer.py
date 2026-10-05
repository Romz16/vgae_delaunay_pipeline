
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
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, float]:
        set_seed(seed)
        data = apply_node_split(data, seed=seed, config=self.config.split, device=self.device)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        in_channels = int(data.num_features)
        out_channels = int(data.y.max().item() + 1)

        model = build_gnn(backbone, in_channels, out_channels, params, self.device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        scheduler = self._build_scheduler(optimizer, params, epochs)

        selection_metric = self.config.training.selection_metric.lower()
        if selection_metric not in {"f1", "acc"}:
            raise ValueError(f"selection_metric deve ser 'f1' ou 'acc', recebido: {selection_metric}")

        best_state: dict[str, torch.Tensor] | None = None
        best_selection_value = -1.0
        best_val_acc = -1.0
        best_val_f1 = -1.0
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_loss = 0.0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.x, edge_index, edge_weight=edge_weight)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                logits = model(data.x, edge_index, edge_weight=edge_weight)
                preds = logits.argmax(dim=1)
                val_metrics = classification_metrics(
                    data.y[data.val_mask].detach().cpu().numpy(),
                    preds[data.val_mask].detach().cpu().numpy(),
                )
                current_selection = val_metrics[selection_metric]
                if current_selection > best_selection_value:
                    best_selection_value = current_selection
                    best_val_acc = val_metrics["acc"]
                    best_val_f1 = val_metrics["f1"]
                    best_loss = float(loss.detach().cpu().item())
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
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
            "best_epoch": float(best_epoch),
            "selection_metric": selection_metric,
            "selection_value": best_selection_value,
        }

    def train_single_run_nested(
        self,
        data: Data,
        backbone: str,
        edge_index: torch.Tensor,
        params: GNNParams,
        seed: int,
        epochs: int,
        edge_weight: torch.Tensor | None = None,
    ) -> dict[str, float]:
        required = ("train_mask", "inner_val_mask", "topology_val_mask", "test_mask")
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(f"Nested split masks are missing: {', '.join(missing)}")

        set_seed(seed)
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        model = build_gnn(
            backbone,
            int(data.num_features),
            int(data.y.max().item() + 1),
            params,
            self.device,
        )
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        scheduler = self._build_scheduler(optimizer, params, epochs)
        selection_metric = self.config.training.selection_metric.lower()
        if selection_metric not in {"f1", "acc"}:
            raise ValueError(f"selection_metric must be 'f1' or 'acc', got {selection_metric}")

        best_state: dict[str, torch.Tensor] | None = None
        best_value = -1.0
        best_epoch = 0
        best_loss = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data.x, edge_index, edge_weight=edge_weight)
            loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                predictions = model(data.x, edge_index, edge_weight=edge_weight).argmax(dim=1)
            inner_metrics = classification_metrics(
                data.y[data.inner_val_mask].detach().cpu().numpy(),
                predictions[data.inner_val_mask].detach().cpu().numpy(),
            )
            if inner_metrics[selection_metric] > best_value:
                best_value = inner_metrics[selection_metric]
                best_epoch = epoch
                best_loss = float(loss.detach().cpu().item())
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("Downstream GNN did not produce a valid checkpoint.")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions = model(data.x, edge_index, edge_weight=edge_weight).argmax(dim=1)

        result: dict[str, float] = {
            "loss": best_loss,
            "best_epoch": float(best_epoch),
            "selection_value": float(best_value),
        }
        for name in ("inner_val", "topology_val", "test"):
            mask = getattr(data, f"{name}_mask")
            metrics = classification_metrics(
                data.y[mask].detach().cpu().numpy(),
                predictions[mask].detach().cpu().numpy(),
            )
            result[f"{name}_acc"] = metrics["acc"]
            result[f"{name}_f1"] = metrics["f1"]
        return result

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
        val_accs: list[float] = []
        val_f1s: list[float] = []
        raw_runs: list[dict[str, object]] = []
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
            val_accs.append(float(metrics["val_acc"]))
            val_f1s.append(float(metrics["val_f1"]))
            raw_runs.append({"seed": seed, **metrics})

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "accs": accs,
            "f1s": f1s,
            "losses": losses,
            "val_accs": val_accs,
            "val_f1s": val_f1s,
            "raw_runs": raw_runs,
            "acc_summary": metric_summary(accs),
            "f1_summary": metric_summary(f1s),
            "loss_summary": metric_summary(losses),
            "val_acc_summary": metric_summary(val_accs),
            "val_f1_summary": metric_summary(val_f1s),
            "elapsed_seconds": time.time() - start,
        }

    def evaluate_validation_selected_rewiring(
        self,
        data: Data,
        backbone: str,
        original_edge_index: torch.Tensor,
        rewired_graphs: dict[float, torch.Tensor],
        params: GNNParams,
        num_runs: int,
        epochs: int,
        label_fn,
    ) -> dict[str, object]:
        all_runs: list[dict[str, object]] = []
        selected_runs: list[dict[str, object]] = []
        baseline_f1s: list[float] = []
        selected_f1s: list[float] = []
        gain_f1s: list[float] = []
        baseline_accs: list[float] = []
        selected_accs: list[float] = []
        gain_accs: list[float] = []
        losses: list[float] = []
        start = time.time()
        selection_metric = self.config.training.selection_metric.lower()
        selection_col = f"val_{selection_metric}"

        for run_idx in range(num_runs):
            seed = self.config.run_seed_start + run_idx
            baseline = self.train_single_run(
                data=data,
                backbone=backbone,
                edge_index=original_edge_index,
                params=params,
                seed=seed,
                epochs=epochs,
                track_test=True,
            )
            baseline_f1 = float(baseline["test_f1"])
            baseline_acc = float(baseline["test_acc"])
            baseline_f1s.append(baseline_f1)
            baseline_accs.append(baseline_acc)
            all_runs.append(
                {
                    "dataset": self.config.dataset_name,
                    "backbone": backbone,
                    "seed": seed,
                    "condition": "Baseline (Original)",
                    "ratio": np.nan,
                    "is_baseline": True,
                    **baseline,
                }
            )

            candidates: list[dict[str, object]] = []
            for ratio, edge_index in sorted(rewired_graphs.items(), key=lambda item: item[0]):
                label = label_fn(ratio)
                metrics = self.train_single_run(
                    data=data,
                    backbone=backbone,
                    edge_index=edge_index,
                    params=params,
                    seed=seed,
                    epochs=epochs,
                    track_test=True,
                )
                row = {
                    "dataset": self.config.dataset_name,
                    "backbone": backbone,
                    "seed": seed,
                    "condition": label,
                    "ratio": float(ratio),
                    "is_baseline": False,
                    **metrics,
                }
                candidates.append(row)
                all_runs.append(row)

            # Stable tie-breaking: highest validation metric, then lower ratio.
            selected = sorted(candidates, key=lambda r: (-float(r[selection_col]), float(r["ratio"])))[0]
            selected_f1 = float(selected["test_f1"])
            selected_acc = float(selected["test_acc"])
            selected_f1s.append(selected_f1)
            selected_accs.append(selected_acc)
            gain_f1s.append(selected_f1 - baseline_f1)
            gain_accs.append(selected_acc - baseline_acc)
            losses.append(float(selected["loss"]))

            selected_runs.append(
                {
                    "dataset": self.config.dataset_name,
                    "backbone": backbone,
                    "seed": seed,
                    "selected_ratio": float(selected["ratio"]),
                    "selected_condition": selected["condition"],
                    "selection_metric": selection_metric,
                    "selected_val_acc": float(selected["val_acc"]),
                    "selected_val_f1": float(selected["val_f1"]),
                    "selected_test_acc": selected_acc,
                    "selected_test_f1": selected_f1,
                    "baseline_val_acc": float(baseline["val_acc"]),
                    "baseline_val_f1": float(baseline["val_f1"]),
                    "baseline_test_acc": baseline_acc,
                    "baseline_test_f1": baseline_f1,
                    "gain_acc": selected_acc - baseline_acc,
                    "gain_f1": selected_f1 - baseline_f1,
                }
            )

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "all_runs": all_runs,
            "selected_runs": selected_runs,
            "baseline_acc_summary": metric_summary(baseline_accs),
            "baseline_f1_summary": metric_summary(baseline_f1s),
            "selected_acc_summary": metric_summary(selected_accs),
            "selected_f1_summary": metric_summary(selected_f1s),
            "gain_acc_summary": metric_summary(gain_accs),
            "gain_f1_summary": metric_summary(gain_f1s),
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
