"""Optuna optimization for node-classification GNN backbones."""

from __future__ import annotations

import logging

import optuna
import torch
from torch_geometric.data import Data

from config import PipelineConfig
from src.models.gnn import GNNParams
from src.training.gnn_trainer import GNNTrainer

LOGGER = logging.getLogger("vgae_delaunay_pipeline.gnn_optimizer")


class GNNOptimizer:
    """Optimize GNN hyperparameters using validation performance."""

    def __init__(self, config: PipelineConfig, device: torch.device) -> None:
        """Initialize the optimizer.

        Args:
            config: Global pipeline configuration.
            device: Target device.
        """
        self.config = config
        self.device = device
        self.trainer = GNNTrainer(config, device)

    def optimize_all(self, data: Data, edge_index: torch.Tensor) -> dict[str, GNNParams]:
        """Optimize all configured backbones.

        Args:
            data: Input graph.
            edge_index: Edge index used during hyperparameter search.

        Returns:
            Mapping from backbone name to optimized hyperparameters.
        """
        best: dict[str, GNNParams] = {}
        for backbone in self.config.backbones:
            best[backbone] = self.optimize_one(data, edge_index, backbone)
        return best

    def optimize_one(self, data: Data, edge_index: torch.Tensor, backbone: str) -> GNNParams:
        """Optimize one GNN backbone.

        Args:
            data: Input graph.
            edge_index: Edge index used for message passing during search.
            backbone: Backbone name.

        Returns:
            Best hyperparameter configuration.
        """
        LOGGER.info("Iniciando busca de hiperparâmetros para %s...", backbone.upper())
        sampler = optuna.samplers.TPESampler(
            seed=self.config.base_seed,
            n_startup_trials=self.config.optuna.n_startup_trials,
        )
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=max(10, self.config.optuna.pruning_warmup_steps // 2),
        )
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            lambda trial: self._objective(trial, data, edge_index, backbone),
            n_trials=self.config.optuna.gnn_trials,
            timeout=self.config.optuna.timeout_seconds,
            show_progress_bar=False,
        )
        LOGGER.info("Melhor score %s: %.4f", backbone.upper(), study.best_value)
        LOGGER.info("Melhores parâmetros %s: %s", backbone.upper(), study.best_params)
        return self._params_from_trial_dict(study.best_params, backbone)

    def _objective(self, trial: optuna.Trial, data: Data, edge_index: torch.Tensor, backbone: str) -> float:
        """Optuna objective for one GNN backbone."""
        params = GNNParams(
            hidden_channels=trial.suggest_categorical("hidden_channels", [32, 64, 128, 256]),
            n_layers=trial.suggest_int("n_layers", 2, 4),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.7),
            activation=trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "gelu"]),
            batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
            scheduler=trial.suggest_categorical("scheduler", ["none", "step", "cosine"]),
            heads=trial.suggest_categorical("heads", [2, 4, 8]) if backbone.lower() == "gat" else None,
        )
        metrics = self.trainer.train_single_run(
            data=data,
            backbone=backbone,
            edge_index=edge_index,
            params=params,
            seed=self.config.base_seed + trial.number,
            epochs=self.config.training.gnn_opt_epochs,
            track_test=False,
        )
        score = 0.5 * float(metrics["val_acc"]) + 0.5 * float(metrics["val_f1"])
        trial.report(score, step=self.config.training.gnn_opt_epochs)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    @staticmethod
    def _params_from_trial_dict(params: dict[str, object], backbone: str) -> GNNParams:
        """Convert Optuna params into a typed dataclass."""
        return GNNParams(
            hidden_channels=int(params["hidden_channels"]),
            n_layers=int(params["n_layers"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            dropout=float(params["dropout"]),
            activation=str(params["activation"]),
            batch_norm=bool(params["batch_norm"]),
            scheduler=str(params["scheduler"]),
            heads=int(params["heads"]) if backbone.lower() == "gat" and "heads" in params else None,
        )
