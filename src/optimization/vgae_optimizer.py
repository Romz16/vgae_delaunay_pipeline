
from __future__ import annotations

import copy
import logging

import optuna
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from config import PipelineConfig
from src.models.vgae import VGAEParams, build_vgae
from src.utils.seed import set_seed

LOGGER = logging.getLogger("vgae_delaunay_pipeline.vgae_optimizer")


class VGAEOptimizer:

    def __init__(self, config: PipelineConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    def optimize(self, data: Data) -> VGAEParams:
        LOGGER.info("Iniciando busca de hiperparâmetros para VGAE...")
        sampler = optuna.samplers.TPESampler(
            seed=self.config.base_seed,
            n_startup_trials=self.config.optuna.n_startup_trials,
        )
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=self.config.optuna.pruning_warmup_steps,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            lambda trial: self._objective(trial, data),
            n_trials=self.config.optuna.vgae_trials,
            timeout=self.config.optuna.timeout_seconds,
            show_progress_bar=False,
        )
        LOGGER.info("Melhor score VGAE: %.4f", study.best_value)
        LOGGER.info("Melhores parâmetros VGAE: %s", study.best_params)
        return self._params_from_trial_dict(study.best_params)

    def _objective(self, trial: optuna.Trial, data: Data) -> float:
        params = VGAEParams(
            hidden_channels=trial.suggest_categorical("hidden_channels", [32, 64, 128, 256]),
            latent_channels=trial.suggest_categorical("latent_channels", [16, 32, 64, 128]),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            dropout=trial.suggest_float("dropout", 0.0, 0.7),
        )
        set_seed(self.config.base_seed + trial.number)
        split_transform = RandomLinkSplit(
            num_val=0.10,
            num_test=0.0,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=False,
        )
        # PyG's ``Data.cpu()`` mutates the object in place.  Clone first so an
        # Optuna trial cannot move the pipeline's shared graph off CUDA.
        split_data = copy.deepcopy(data).detach().cpu()
        train_data, val_data, _ = split_transform(split_data)
        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)

        model = build_vgae(int(data.num_features), params, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

        for epoch in range(1, self.config.training.vgae_opt_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(train_data.x, train_data.edge_index)
            loss = model.recon_loss(z, train_data.edge_index)
            loss = loss + (1.0 / train_data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                model.eval()
                with torch.no_grad():
                    z = model.encode(train_data.x, train_data.edge_index)
                    auc, ap = model.test(
                        z,
                        val_data.pos_edge_label_index,
                        val_data.neg_edge_label_index,
                    )
                    score = float((auc + ap) / 2.0)
                trial.report(score, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_index)
            auc, ap = model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)
        return float((auc + ap) / 2.0)

    @staticmethod
    def _params_from_trial_dict(params: dict[str, object]) -> VGAEParams:
        return VGAEParams(
            hidden_channels=int(params["hidden_channels"]),
            latent_channels=int(params["latent_channels"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            dropout=float(params["dropout"]),
        )
