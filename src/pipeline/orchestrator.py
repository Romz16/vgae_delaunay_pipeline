from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config import PipelineConfig
from src.data.splits import apply_node_split
from src.models.gnn import GNNParams
from src.models.vgae import VGAEParams
from src.optimization.gnn_optimizer import GNNOptimizer
from src.optimization.vgae_optimizer import VGAEOptimizer
from src.reporting.report_writer import ReportWriter
from src.rewiring.delaunay_builder import DelaunayGraphBuilder
from src.rewiring.vgae_delaunay import VGAEDelaunayRewirer
from src.training.gnn_trainer import GNNTrainer
from src.training.vgae_trainer import VGAETrainer
from src.utils.seed import set_seed

LOGGER = logging.getLogger("vgae_delaunay_pipeline.orchestrator")


class VGAEDelaunayPipeline:
    """Orchestrate the full scientific pipeline.

    Pipeline:
        1. Optimize VGAE hyperparameters.
        2. Train final VGAE and save embeddings as ``.npy``.
        3. Optimize GCN/GAT/GraphSAGE/GCN_Residual hyperparameters.
        4. Build DGlf with auxiliary GCN -> UMAP -> Delaunay.
        5. Apply VGAE-guided Delaunay rewiring for all ratios.
        6. Evaluate all backbones and export a CSV/Markdown report.
    """

    def __init__(self, config: PipelineConfig, device: torch.device) -> None:

        self.config = config
        self.device = device
        self.report_writer = ReportWriter()

    def run(self, data: Data) -> pd.DataFrame:

        set_seed(self.config.base_seed)
        self._prepare_output_dirs()
        self._save_config()

        LOGGER.info("Dataset: %s", self.config.dataset_name)
        LOGGER.info(
            "Grafo | Nós: %d | Arestas PyG: %d | Features: %d | Classes: %d",
            data.num_nodes,
            data.edge_index.size(1),
            data.num_features,
            int(data.y.max().item() + 1),
        )

        LOGGER.info("Etapa 1/6 | Otimizando VGAE...")
        vgae_params = VGAEOptimizer(self.config, self.device).optimize(data)
        self._save_params("best_vgae_params.json", asdict(vgae_params))

        LOGGER.info("Etapa 2/6 | Treinando VGAE final e salvando embeddings...")
        embeddings = self._fit_and_save_vgae(data, vgae_params)

        LOGGER.info("Etapa 3/6 | Otimizando hiperparâmetros dos modelos GNN...")
        gnn_params = GNNOptimizer(self.config, self.device).optimize_all(
            data, data.edge_index
        )
        self._save_params(
            "best_gnn_params.json", {k: asdict(v) for k, v in gnn_params.items()}
        )

        LOGGER.info("Etapa 4/6 | Gerando DGlf via GCN auxiliar -> UMAP -> Delaunay...")
        # Fixa uma split para o pré-treino do DGlf, mantendo o desenho original.
        data_for_delaunay = apply_node_split(
            data,
            seed=self.config.base_seed,
            config=self.config.split,
            device=self.device,
        )
        edge_index_delaunay = DelaunayGraphBuilder(self.config, self.device).build(
            data_for_delaunay
        )

        LOGGER.info("Etapa 5/6 | Aplicando rewiring DGlf + VGAE para todas as taxas...")
        rewired_graphs = self._build_rewired_graphs(
            edge_index_delaunay, embeddings, int(data.num_nodes)
        )

        LOGGER.info("Etapa 6/6 | Avaliando baseline e grafos rewired...")
        rows = self._evaluate_all(data, gnn_params, rewired_graphs)
        df = self.report_writer.export(
            rows, self.config.result_csv_path, self.config.result_md_path
        )
        LOGGER.info("CSV final salvo em: %s", self.config.result_csv_path)
        LOGGER.info("Markdown final salvo em: %s", self.config.result_md_path)
        return df

    def _fit_and_save_vgae(self, data: Data, params: VGAEParams) -> np.ndarray:

        trainer = VGAETrainer(self.device)
        model = trainer.train(
            data=data,
            params=params,
            epochs=self.config.training.vgae_final_epochs,
            seed=self.config.base_seed,
        )
        embeddings = trainer.extract_embeddings(model, data)
        self.config.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.config.embedding_path, embeddings)
        LOGGER.info(
            "Embeddings salvos em: %s | Shape: %s",
            self.config.embedding_path,
            embeddings.shape,
        )
        return embeddings

    def _build_rewired_graphs(
        self,
        edge_index_delaunay: torch.Tensor,
        embeddings: np.ndarray,
        num_nodes: int,
    ) -> dict[float, torch.Tensor]:
        """Create one rewired graph per ratio."""
        rewirer = VGAEDelaunayRewirer(
            add_self_loops=self.config.add_self_loops_after_rewire,
            device=self.device,
        )
        rewired: dict[float, torch.Tensor] = {}
        for ratio in self.config.rewiring_ratios:
            LOGGER.info("Aplicando DGlf + VGAE %.0f%%...", ratio * 100)
            rewired[ratio] = rewirer.rewire(
                edge_index=edge_index_delaunay,
                vgae_embeddings=embeddings,
                ratio=ratio,
                num_nodes=num_nodes,
            )
        return rewired

    def _evaluate_all(
        self,
        data: Data,
        params_by_backbone: dict[str, GNNParams],
        rewired_graphs: dict[float, torch.Tensor],
    ) -> list[dict[str, object]]:
        """Evaluate baseline and all rewiring conditions."""
        trainer = GNNTrainer(self.config, self.device)
        rows: list[dict[str, object]] = []

        for backbone, params in params_by_backbone.items():
            LOGGER.info("Baseline para %s...", backbone.upper())
            baseline = trainer.evaluate_repeated_runs(
                data=data,
                backbone=backbone,
                edge_index=data.edge_index,
                params=params,
                num_runs=self.config.training.final_runs,
                epochs=self.config.training.gnn_final_epochs,
            )
            rows.append(
                ReportWriter.make_row(backbone, "Baseline (Original)", baseline)
            )

        for backbone, params in params_by_backbone.items():
            for ratio, edge_index in rewired_graphs.items():
                label = f"DGlf + VGAE {int(ratio * 100)}%"
                LOGGER.info("%s para %s...", label, backbone.upper())
                result = trainer.evaluate_repeated_runs(
                    data=data,
                    backbone=backbone,
                    edge_index=edge_index,
                    params=params,
                    num_runs=self.config.training.final_runs,
                    epochs=self.config.training.gnn_final_epochs,
                )
                rows.append(ReportWriter.make_row(backbone, label, result))

        return rows

    def _prepare_output_dirs(self) -> None:
        """Create all output directories."""
        self.config.output.embeddings_dir().mkdir(parents=True, exist_ok=True)
        self.config.output.results_dir().mkdir(parents=True, exist_ok=True)
        self.config.output.studies_dir().mkdir(parents=True, exist_ok=True)

    def _save_config(self) -> None:
        """Persist the pipeline configuration as JSON."""
        self._save_params("pipeline_config.json", asdict(self.config))

    def _save_params(self, filename: str, payload: dict[str, object]) -> None:
        """Save a dictionary as JSON in the output directory."""
        path = self.config.output.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self._json_safe(payload), fp, indent=2, ensure_ascii=False)

    def _json_safe(self, value):
        """Convert Paths and nested objects into JSON-safe values."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value
