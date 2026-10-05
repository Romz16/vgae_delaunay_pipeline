
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
from src.analysis.statistical_tests import paired_test_rows
from src.analysis.structural_correlations import (
    baseline_saturation_analysis,
    dataset_level_gain,
    density_confounder_analysis,
    exploratory_recommendations,
    leave_one_out_sensitivity,
    old_vs_corrected,
    structural_metric_correlations,
)
from src.analysis.structural_metrics import (
    StructuralMetricConfig,
    compute_structural_metrics,
    save_structural_metrics,
)
from src.data.splits import apply_node_split
from src.models.gnn import GNNParams
from src.models.vgae import VGAEParams
from src.optimization.gnn_optimizer import GNNOptimizer
from src.optimization.vgae_optimizer import VGAEOptimizer
from src.reporting.report_writer import ReportWriter
from src.rewiring.delaunay_builder import DelaunayGraphBuilder
from src.rewiring.hybrid_vgae_ct import HybridVGAECTDelaunayRewirer
from src.rewiring.vgae_delaunay import VGAEDelaunayRewirer
from src.training.gnn_trainer import GNNTrainer
from src.training.vgae_trainer import VGAETrainer
from src.utils.seed import set_seed

LOGGER = logging.getLogger("vgae_delaunay_pipeline.orchestrator")


class VGAEDelaunayPipeline:

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

        structural_cfg = StructuralMetricConfig(seed=self.config.base_seed)
        original_metric_row = compute_structural_metrics(
            data=data,
            dataset=self.config.dataset_name,
            graph_name="Original",
            config=structural_cfg,
        )
        save_structural_metrics([original_metric_row], self._revision_path("original_graph_metrics.csv"))

        LOGGER.info("Etapa 1/6 | Otimizando VGAE...")
        vgae_params = VGAEOptimizer(self.config, self.device).optimize(data)
        self._save_params("best_vgae_params.json", asdict(vgae_params))

        LOGGER.info("Etapa 2/6 | Treinando VGAE final e salvando embeddings...")
        embeddings = self._fit_and_save_vgae(data, vgae_params)

        LOGGER.info("Etapa 3/6 | Otimizando hiperparâmetros dos modelos GNN...")
        gnn_params = GNNOptimizer(self.config, self.device).optimize_all(data, data.edge_index)
        self._save_params("best_gnn_params.json", {k: asdict(v) for k, v in gnn_params.items()})

        LOGGER.info("Etapa 4/6 | Gerando DGlf via GCN auxiliar -> UMAP -> Delaunay...")
        data_for_delaunay = apply_node_split(
            data,
            seed=self.config.base_seed,
            config=self.config.split,
            device=self.device,
        )
        builder = DelaunayGraphBuilder(self.config, self.device)
        edge_index_delaunay, learned_features, points_2d = builder.build_with_projection(data_for_delaunay)
        LOGGER.info("Gerando baseline Raw features -> UMAP -> Delaunay...")
        raw_edge_index, raw_features, raw_points_2d = builder.build_from_raw_features_with_projection(data_for_delaunay)
        self._write_umap_trustworthiness(
            builder,
            [
                ("auxiliary_gcn_first_layer", learned_features, points_2d),
                ("raw_features", raw_features, raw_points_2d),
            ],
        )

        LOGGER.info("Etapa 5/6 | Aplicando rewiring DGlf + VGAE para todas as taxas...")
        rewired_graphs = self._build_rewired_graphs(edge_index_delaunay, embeddings, int(data.num_nodes))
        raw_metric_row = compute_structural_metrics(
            data=data,
            dataset=self.config.dataset_name,
            graph_name="Raw features -> UMAP -> Delaunay",
            config=structural_cfg,
            edge_index=raw_edge_index,
        )
        all_structural_rows = [original_metric_row, raw_metric_row]
        for ratio, edge_index in rewired_graphs.items():
            all_structural_rows.append(
                compute_structural_metrics(
                    data=data,
                    dataset=self.config.dataset_name,
                    graph_name=self._rewiring_label(ratio),
                    config=structural_cfg,
                    edge_index=edge_index,
                )
            )
        graph_metrics_df = save_structural_metrics(
            all_structural_rows,
            self._revision_path("selected_rewired_graph_metrics.csv"),
        )

        LOGGER.info("Etapa 6/6 | Avaliando baseline e selecionando r por validação...")
        if self.config.training.validation_select_rewiring:
            final_df = self._evaluate_validation_selected(data, gnn_params, rewired_graphs, graph_metrics_df)
        else:
            LOGGER.warning("Executando protocolo legado: todas as taxas serão avaliadas sem seleção por validação.")
            rows = self._evaluate_legacy_all_rates(data, gnn_params, rewired_graphs)
            final_df = self.report_writer.export(rows, self.config.result_csv_path, self.config.result_md_path)

        LOGGER.info("Avaliando baseline Raw features -> UMAP -> Delaunay...")
        self._evaluate_raw_feature_baseline(data, gnn_params, raw_edge_index)

        self._write_static_revision_docs()
        LOGGER.info("Resultados oficiais salvos em: %s", self._revision_path("final_selected_test_results.csv"))
        return final_df

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
        LOGGER.info("Embeddings salvos em: %s | Shape: %s", self.config.embedding_path, embeddings.shape)
        return embeddings

    def _build_rewired_graphs(
        self,
        edge_index_delaunay: torch.Tensor,
        embeddings: np.ndarray,
        num_nodes: int,
    ) -> dict[float, torch.Tensor]:
        strategy = self.config.rewiring.strategy.lower()
        if strategy == "hybrid_ct":
            rewirer = HybridVGAECTDelaunayRewirer(
                config=self.config.rewiring,
                add_self_loops=self.config.add_self_loops_after_rewire,
                device=self.device,
            )
        elif strategy == "vgae":
            rewirer = VGAEDelaunayRewirer(
                add_self_loops=self.config.add_self_loops_after_rewire,
                device=self.device,
            )
        else:
            raise ValueError(f"Estratégia de rewiring desconhecida: {self.config.rewiring.strategy}")

        rewired: dict[float, torch.Tensor] = {}
        for ratio in self.config.rewiring_ratios:
            LOGGER.info("Aplicando %s %.0f%%...", self.config.rewiring.strategy, ratio * 100)
            rewired[ratio] = rewirer.rewire(
                edge_index=edge_index_delaunay,
                vgae_embeddings=embeddings,
                ratio=ratio,
                num_nodes=num_nodes,
            )
        return rewired

    def _evaluate_validation_selected(
        self,
        data: Data,
        params_by_backbone: dict[str, GNNParams],
        rewired_graphs: dict[float, torch.Tensor],
        graph_metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        trainer = GNNTrainer(self.config, self.device)
        summary_rows: list[dict[str, object]] = []
        all_runs: list[dict[str, object]] = []
        selected_runs: list[dict[str, object]] = []

        for backbone, params in params_by_backbone.items():
            LOGGER.info("Seleção validation-only de r para %s...", backbone.upper())
            result = trainer.evaluate_validation_selected_rewiring(
                data=data,
                backbone=backbone,
                original_edge_index=data.edge_index,
                rewired_graphs=rewired_graphs,
                params=params,
                num_runs=self.config.training.final_runs,
                epochs=self.config.training.gnn_final_epochs,
                label_fn=self._rewiring_label,
            )
            summary_rows.append(
                {
                    "dataset": self.config.dataset_name,
                    **ReportWriter.make_selected_summary_row(backbone, result),
                }
            )
            all_runs.extend(result["all_runs"])
            selected_runs.extend(result["selected_runs"])

        all_runs_df = pd.DataFrame(all_runs)
        selected_runs_df = pd.DataFrame(selected_runs)
        final_df = pd.DataFrame(summary_rows)
        all_runs_df.to_csv(self._revision_path("all_runs_long.csv"), index=False)
        selected_runs_df.to_csv(self._revision_path("selected_rates.csv"), index=False)
        final_df.to_csv(self._revision_path("final_selected_test_results.csv"), index=False)
        final_df.to_csv(self.config.result_csv_path, index=False)

        self._write_confidence_intervals(final_df)
        paired = paired_test_rows(selected_runs_df)
        paired.to_csv(self._revision_path("paired_statistical_tests.csv"), index=False)

        correlations = structural_metric_correlations(graph_metrics_df, selected_runs_df)
        correlations.to_csv(self._revision_path("structural_metric_correlations.csv"), index=False)
        sensitivity = leave_one_out_sensitivity(graph_metrics_df, selected_runs_df)
        sensitivity.to_csv(self._revision_path("structural_correlation_sensitivity.csv"), index=False)
        density_confounder_analysis(graph_metrics_df, selected_runs_df).to_csv(
            self._revision_path("density_confounder_analysis.csv"), index=False
        )
        baseline_saturation_analysis(graph_metrics_df, selected_runs_df).to_csv(
            self._revision_path("baseline_saturation_analysis.csv"), index=False
        )
        old_vs_corrected(self._preliminary_correlations(), correlations).to_csv(
            self._revision_path("structural_correlations_old_vs_corrected.csv"), index=False
        )
        dataset_gain = dataset_level_gain(selected_runs_df)
        exploratory_recommendations(graph_metrics_df, dataset_gain).to_csv(
            self._revision_path("exploratory_rewiring_recommendations.csv"), index=False
        )
        self._write_figures(graph_metrics_df, selected_runs_df)
        return final_df

    def _evaluate_legacy_all_rates(
        self,
        data: Data,
        params_by_backbone: dict[str, GNNParams],
        rewired_graphs: dict[float, torch.Tensor],
    ) -> list[dict[str, object]]:
        trainer = GNNTrainer(self.config, self.device)
        rows: list[dict[str, object]] = []
        for backbone, params in params_by_backbone.items():
            baseline = trainer.evaluate_repeated_runs(
                data=data,
                backbone=backbone,
                edge_index=data.edge_index,
                params=params,
                num_runs=self.config.training.final_runs,
                epochs=self.config.training.gnn_final_epochs,
            )
            rows.append(ReportWriter.make_row(backbone, "Baseline (Original)", baseline))
            for ratio, edge_index in rewired_graphs.items():
                result = trainer.evaluate_repeated_runs(
                    data=data,
                    backbone=backbone,
                    edge_index=edge_index,
                    params=params,
                    num_runs=self.config.training.final_runs,
                    epochs=self.config.training.gnn_final_epochs,
                )
                rows.append(ReportWriter.make_row(backbone, self._rewiring_label(ratio), result))
        return rows

    def _rewiring_label(self, ratio: float) -> str:
        if self.config.rewiring.strategy.lower() == "hybrid_ct":
            alpha = self.config.rewiring.hybrid_alpha
            return f"DGlf + VGAE+CT a={alpha:.2f} {int(ratio * 100)}%"
        return f"DGlf + VGAE {int(ratio * 100)}%"

    def _prepare_output_dirs(self) -> None:
        self.config.output.embeddings_dir().mkdir(parents=True, exist_ok=True)
        self.config.output.results_dir().mkdir(parents=True, exist_ok=True)
        self.config.output.studies_dir().mkdir(parents=True, exist_ok=True)
        self.config.output.revision_dir().mkdir(parents=True, exist_ok=True)
        self.config.output.figures_dir().mkdir(parents=True, exist_ok=True)

    def _revision_path(self, filename: str) -> Path:
        return self.config.output.revision_dir() / filename

    def _save_config(self) -> None:
        self._save_params("pipeline_config.json", asdict(self.config))

    def _save_params(self, filename: str, payload: dict[str, object]) -> None:
        path = self.config.output.root_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self._json_safe(payload), fp, indent=2, ensure_ascii=False)

    def _write_umap_trustworthiness(
        self,
        builder: DelaunayGraphBuilder,
        projections: list[tuple[str, np.ndarray, np.ndarray]],
        append: bool = False,
    ) -> None:
        rows: list[dict[str, object]] = []
        for source_representation, features, points_2d in projections:
            try:
                trust = builder.umap_trustworthiness(features, points_2d)
            except Exception as exc:
                LOGGER.warning("Não foi possível calcular UMAP trustworthiness para %s: %s", source_representation, exc)
                trust = np.nan
            rows.append(
                {
                    "dataset": self.config.dataset_name,
                    "source_representation": source_representation,
                    "projection": "umap_2d",
                    "trustworthiness": trust,
                    "num_samples_total": int(features.shape[0]),
                    "num_samples_used": min(
                        int(features.shape[0]),
                        int(self.config.delaunay.trustworthiness_max_samples),
                    ),
                    "sampling_seed": self.config.delaunay.umap_seed,
                    "umap_neighbors": self.config.delaunay.umap_neighbors,
                    "umap_components": self.config.delaunay.umap_components,
                }
            )
        frame = pd.DataFrame(rows)
        path = self._revision_path("umap_trustworthiness.csv")
        if append and path.exists():
            previous = pd.read_csv(path)
            keys = set(frame["source_representation"])
            previous = previous[~previous["source_representation"].isin(keys)]
            frame = pd.concat([previous, frame], ignore_index=True)
        frame.to_csv(path, index=False)

    def _evaluate_raw_feature_baseline(
        self,
        data: Data,
        params_by_backbone: dict[str, GNNParams],
        raw_edge_index: torch.Tensor,
    ) -> pd.DataFrame:
        trainer = GNNTrainer(self.config, self.device)
        summary_rows: list[dict[str, object]] = []
        all_runs: list[dict[str, object]] = []
        for backbone, params in params_by_backbone.items():
            result = trainer.evaluate_repeated_runs(
                data=data,
                backbone=backbone,
                edge_index=raw_edge_index,
                params=params,
                num_runs=self.config.training.final_runs,
                epochs=self.config.training.gnn_final_epochs,
            )
            summary_rows.append(
                {
                    "dataset": self.config.dataset_name,
                    "Protocol": "raw_features_umap_delaunay",
                    **ReportWriter.make_row(backbone, "Raw features -> UMAP -> Delaunay", result),
                }
            )
            all_runs.extend(
                {
                    "dataset": self.config.dataset_name,
                    "backbone": backbone,
                    "condition": "Raw features -> UMAP -> Delaunay",
                    **run,
                }
                for run in result["raw_runs"]
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self._revision_path("raw_feature_delaunay_results.csv"), index=False)
        pd.DataFrame(all_runs).to_csv(self._revision_path("raw_feature_delaunay_all_runs.csv"), index=False)
        return summary

    def _write_confidence_intervals(self, final_df: pd.DataFrame) -> None:
        cols = [
            "Modelo",
            "baseline_f1_mean",
            "selected_f1_mean",
            "gain_f1_mean",
            "gain_f1_ci95_lower",
            "gain_f1_ci95_upper",
            "gain_acc_mean",
            "gain_acc_ci95_lower",
            "gain_acc_ci95_upper",
        ]
        final_df[[c for c in cols if c in final_df.columns]].to_csv(
            self._revision_path("confidence_intervals.csv"), index=False
        )

    def _write_figures(self, graph_metrics_df: pd.DataFrame, selected_runs_df: pd.DataFrame) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            LOGGER.warning("matplotlib indisponível; figuras não serão geradas.")
            return
        dataset_gain = dataset_level_gain(selected_runs_df)
        original = graph_metrics_df[graph_metrics_df["graph_name"].eq("Original")]
        merged = dataset_gain.merge(original, on="dataset", how="left")
        plots = [
            ("cheeger_upper_bound", "corrected_mean_gain", "cheeger_upper_vs_corrected_gain.png"),
            ("lambda2_norm_laplacian", "corrected_mean_gain", "lambda2_vs_corrected_gain.png"),
            ("density", "corrected_mean_gain", "density_vs_corrected_gain.png"),
            ("edge_homophily", "baseline_f1_mean", "homophily_vs_baseline.png"),
            ("baseline_f1_mean", "corrected_mean_gain", "baseline_vs_corrected_gain.png"),
        ]
        for x, y, filename in plots:
            if x not in merged.columns or y not in merged.columns:
                continue
            frame = merged[[x, y, "dataset"]].replace([np.inf, -np.inf], np.nan).dropna()
            if frame.empty:
                continue
            plt.figure(figsize=(6, 4))
            plt.scatter(frame[x], frame[y])
            for _, row in frame.iterrows():
                plt.annotate(str(row["dataset"]), (row[x], row[y]), fontsize=7)
            plt.xlabel(x)
            plt.ylabel(y)
            plt.tight_layout()
            plt.savefig(self.config.output.figures_dir() / filename, dpi=200)
            plt.close()
        # Placeholder file for delta plots; generated by aggregate script when before/after selected metrics exist.
        (self.config.output.figures_dir() / "delta_structure_vs_gain.README.txt").write_text(
            "Generated by the aggregation step when selected rewired before/after metrics are available.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _preliminary_correlations() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": "lambda2_norm_laplacian",
                    "pearson_r": 0.696,
                    "pearson_p": 0.0119,
                    "spearman_rho": 0.846,
                    "spearman_p": 0.0005,
                },
                {
                    "metric": "cheeger_upper_bound",
                    "pearson_r": 0.760,
                    "pearson_p": 0.0041,
                    "spearman_rho": 0.846,
                    "spearman_p": 0.0005,
                },
                {
                    "metric": "density",
                    "pearson_r": 0.649,
                    "pearson_p": 0.0223,
                    "spearman_rho": 0.783,
                    "spearman_p": 0.0026,
                },
            ]
        )

    def _write_static_revision_docs(self) -> None:
        self._revision_path("dataset_name_mapping.csv").write_text(
            "requested_dataset,loaded_dataset,notes\n"
            f"{self.config.dataset_name},{self.config.dataset_name},Verify aliases in GraphLoader before final publication.\n",
            encoding="utf-8",
        )
        self._revision_path("runtime_complexity_summary.csv").write_text(
            "component,time_complexity,memory_complexity,notes\n"
            "vgae_training,depends_on_epochs_and_edges,model_parameters_and_graph,VGAE optimized/trained before rewiring\n"
            "auxiliary_gcn_umap_delaunay,GCN plus UMAP plus O(n log n),projection_and_graph,Delaunay built from UMAP-projected auxiliary GCN features\n"
            "vgae_similarity,O(N^2 d_z),O(N^2),Full cosine-similarity matrix in vgae_delaunay.py can dominate large graphs\n"
            "candidate_ranking,up_to_O(N^2),candidate_scores,Depends on number of non-edges considered\n",
            encoding="utf-8",
        )
        raw_results_path = self._revision_path("raw_feature_delaunay_results.csv")
        if not raw_results_path.exists():
            raw_results_path.write_text(
                "dataset,status,notes\n"
                f"{self.config.dataset_name},not_run,Raw-feature Delaunay ablation did not complete.\n",
                encoding="utf-8",
            )
        self._revision_path("EXPERIMENT_PROTOCOL.md").write_text(
            "# Corrected Experiment Protocol\n\n"
            "For each dataset, backbone and seed, every candidate rewiring rate is trained on the same train/validation/test split. "
            "The selected rate r* is chosen only by validation performance. The official test result is then the test score of r*. "
            "The pipeline must not report max(test F1 across rates) as the official gain.\n\n"
            "Structural metrics are computed on the original graph immediately after loading and are later merged with validation-selected gains.\n",
            encoding="utf-8",
        )
        self._revision_path("protocol_audit.md").write_text(
            "# Protocol Audit\n\n"
            "Implemented changes:\n\n"
            "- Added validation-only selection of rewiring rate per dataset/backbone/seed.\n"
            "- Added per-seed long-format exports.\n"
            "- Added paired statistical tests between baseline and selected rewiring.\n"
            "- Added original and rewired structural metrics.\n"
            "- Added exploratory structural correlations and sensitivity hooks.\n\n"
            "Scientific note: correlations based on N=12 datasets should be interpreted as exploratory.\n",
            encoding="utf-8",
        )
        self._revision_path("REVISION_EXPERIMENT_SUMMARY.md").write_text(
            "# Revision Experiment Summary\n\n"
            "This file is generated by the revised pipeline. After all datasets are rerun with validation-selected rates, "
            "use the exported CSV files to answer whether lambda2, Cheeger-related proxies, density, homophily and baseline saturation remain associated with corrected gain.\n\n"
            "Do not copy preliminary correlations to the paper without rerunning the corrected protocol.\n",
            encoding="utf-8",
        )
        self._revision_path("CHANGELOG_REVISION.md").write_text(
            "# Changelog Revision\n\n"
            "- Added src/analysis/structural_metrics.py.\n"
            "- Added validation-selected rewiring evaluation in GNNTrainer.\n"
            "- Updated ReportWriter to export CI lower/upper fields.\n"
            "- Updated orchestrator to produce outputs_revision artifacts.\n"
            "- Added UMAP trustworthiness computation for auxiliary-GCN projection.\n",
            encoding="utf-8",
        )

    def _json_safe(self, value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        return value
