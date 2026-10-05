"""Execution engine for controlled rewiring experiments.

The module deliberately keeps graph generation, task generation, pipeline
evaluation, and indicator fitting in separate artifacts. Test outcomes are
written only as targets and are never merged into the prerewiring feature set
used by the indicator.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import networkx as nx
import numpy as np
import pandas as pd

from .config import SyntheticExperimentConfig
from .generation import SyntheticGraph, SyntheticTask, generate_graphs, generate_tasks, stable_seed
from .structural import compute_prerewiring_metrics


def graph_catalog(config: SyntheticExperimentConfig, overwrite: bool = False) -> pd.DataFrame:
    """Generate graph-level prerewiring metrics without labels or model results."""
    output = config.output_dir / "synthetic_graph_metrics.csv"
    if output.exists() and not overwrite:
        return pd.read_csv(output)
    rows: list[dict[str, object]] = []
    for synthetic in generate_graphs(config.graph, config.master_seed):
        metrics = compute_prerewiring_metrics(
            synthetic.graph, config.metrics, stable_seed(config.master_seed, synthetic.graph_id, "metrics")
        )
        rows.append(
            {
                "graph_id": synthetic.graph_id,
                "family": synthetic.specification.family,
                "configuration_id": synthetic.specification.configuration_id,
                "instance_index": synthetic.instance_index,
                "generation_seed": synthetic.generation_seed,
                "generation_parameters": json.dumps(synthetic.specification.parameters, sort_keys=True),
                "generation_metadata": json.dumps(synthetic.generation_metadata, sort_keys=True),
                **{f"param_{key}": value for key, value in synthetic.specification.parameters.items()},
                **metrics,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["family", "configuration_id", "instance_index"])
    _atomic_csv(frame, output)
    return frame


def run_experiments(
    config: SyntheticExperimentConfig,
    *,
    resume: bool = True,
    max_graphs: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    finalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all graph, task, and pipeline seeds with resumable part files."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    _require_pipeline_dependencies()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.output_dir / "experiment_config.json")
    graph_catalog(config)
    parts_dir = config.output_dir / "run_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for path in parts_dir.glob("*.csv"):
            path.unlink()
    selected_ids = {path.name.removesuffix(".selected.csv") for path in parts_dir.glob("*.selected.csv")}
    condition_ids = {path.name.removesuffix(".conditions.csv") for path in parts_dir.glob("*.conditions.csv")}
    delta_ids = {path.name.removesuffix(".deltas.csv") for path in parts_dir.glob("*.deltas.csv")}
    completed = selected_ids & condition_ids & delta_ids if resume else set()
    for graph_index, synthetic in enumerate(generate_graphs(config.graph, config.master_seed)):
        if max_graphs is not None and graph_index >= max_graphs:
            break
        if graph_index % num_shards != shard_index:
            continue
        for task in generate_tasks(synthetic, config.task, config.master_seed):
            for pipeline_index in range(config.pipeline.pipeline_seeds_per_task):
                pipeline_seed = config.pipeline.pipeline_seed_start + pipeline_index
                run_id = f"{task.task_id}__p{pipeline_seed}"
                if run_id in completed:
                    continue
                condition_rows, selected_row, delta_rows = evaluate_one(
                    synthetic, task, pipeline_seed, config
                )
                _atomic_csv(pd.DataFrame(condition_rows), parts_dir / f"{run_id}.conditions.csv")
                _atomic_csv(pd.DataFrame([selected_row]), parts_dir / f"{run_id}.selected.csv")
                _atomic_csv(pd.DataFrame(delta_rows), parts_dir / f"{run_id}.deltas.csv")
    if finalize:
        conditions = _collect_parts(parts_dir, "*.conditions.csv", config.output_dir / "all_execution_results.csv")
        selected = _collect_parts(parts_dir, "*.selected.csv", config.output_dir / "selected_execution_results.csv")
        mechanisms = _collect_parts(parts_dir, "*.deltas.csv", config.output_dir / "postrewiring_mechanism_metrics.csv")
        _postrewiring_deltas(mechanisms, config.output_dir / "postrewiring_deltas.csv")
        aggregate_outcomes(config)
        return conditions, selected
    return pd.DataFrame(), pd.DataFrame()


def evaluate_one(
    synthetic: SyntheticGraph,
    task: SyntheticTask,
    pipeline_seed: int,
    config: SyntheticExperimentConfig,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Evaluate original, geometry, VGAE refinement, and constructor controls."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx, to_undirected

    from config import DelaunayConfig, OptunaConfig, OutputConfig, PipelineConfig, RewiringConfig, SplitConfig, TrainingConfig
    from src.data.splits import apply_node_split
    from src.models.gnn import GNNParams
    from src.models.vgae import VGAEParams
    from src.rewiring.constructor_controls import build_constructor_controls
    from src.rewiring.delaunay_builder import DelaunayGraphBuilder
    from src.rewiring.vgae_delaunay import VGAEDelaunayRewirer
    from src.training.gnn_trainer import GNNTrainer
    from src.training.vgae_trainer import VGAETrainer
    from src.utils.device import resolve_device
    from src.utils.seed import set_seed

    pcfg = config.pipeline
    device = resolve_device("auto")
    set_seed(pipeline_seed)
    nx_data = from_networkx(synthetic.graph)
    data = Data(
        x=torch.tensor(task.features, dtype=torch.float32),
        edge_index=nx_data.edge_index.long(),
        y=torch.tensor(task.labels, dtype=torch.long),
        num_nodes=synthetic.graph.number_of_nodes(),
    ).to(device)
    runtime_dir = config.output_dir / "runtime" / task.task_id / str(pipeline_seed)
    pipeline_config = PipelineConfig(
        dataset_name=task.task_id,
        base_seed=pipeline_seed,
        run_seed_start=pipeline_seed,
        rewiring_ratios=pcfg.rewiring_ratios,
        backbones=(pcfg.downstream_backbone,),
        device="auto",
        split=SplitConfig(),
        rewiring=RewiringConfig(strategy="vgae"),
        optuna=OptunaConfig(vgae_trials=0, gnn_trials=0),
        training=TrainingConfig(
            vgae_opt_epochs=pcfg.vgae_epochs,
            vgae_final_epochs=pcfg.vgae_epochs,
            gnn_opt_epochs=pcfg.downstream_epochs,
            gnn_final_epochs=pcfg.downstream_epochs,
            final_runs=1,
            selection_metric=pcfg.selection_metric,
            validation_select_rewiring=True,
        ),
        delaunay=DelaunayConfig(
            pretrain_epochs=pcfg.auxiliary_epochs,
            pretrain_hidden_channels=pcfg.hidden_channels,
            umap_seed=pipeline_seed,
            umap_neighbors=min(15, max(3, data.num_nodes // 10)),
            trustworthiness_max_samples=data.num_nodes,
        ),
        output=OutputConfig(root_dir=runtime_dir),
    )
    split_data = apply_node_split(data, seed=pipeline_seed, config=pipeline_config.split, device=device)

    vgae_params = VGAEParams(
        hidden_channels=pcfg.hidden_channels,
        latent_channels=pcfg.latent_channels,
        lr=pcfg.learning_rate,
        weight_decay=pcfg.weight_decay,
        dropout=pcfg.dropout,
    )
    gnn_params = GNNParams(
        hidden_channels=pcfg.hidden_channels,
        n_layers=2,
        lr=pcfg.learning_rate,
        weight_decay=pcfg.weight_decay,
        dropout=pcfg.dropout,
        activation="relu",
        batch_norm=True,
        scheduler="none",
        heads=2,
    )
    started = time.perf_counter()
    vgae = VGAETrainer(device)
    vgae_model = vgae.train(data, vgae_params, pcfg.vgae_epochs, pipeline_seed)
    embeddings = vgae.extract_embeddings(vgae_model, data)
    vgae_seconds = time.perf_counter() - started

    started = time.perf_counter()
    builder = DelaunayGraphBuilder(pipeline_config, device)
    delaunay_edges, learned_features, points = builder.build_with_projection(split_data)
    raw_delaunay_edges, raw_features, raw_points = builder.build_from_raw_features_with_projection(split_data)
    learned_trustworthiness = builder.umap_trustworthiness(learned_features, points)
    raw_trustworthiness = builder.umap_trustworthiness(raw_features, raw_points)
    geometry_seconds = time.perf_counter() - started
    controls = build_constructor_controls(points, delaunay_edges, seed=pipeline_seed)
    learned_hd = build_constructor_controls(learned_features, delaunay_edges, seed=pipeline_seed)["knn"]
    raw_hd = build_constructor_controls(raw_features, delaunay_edges, seed=pipeline_seed)["knn"]
    bases = {"delaunay": delaunay_edges}
    for name in pcfg.constructors:
        if name == "raw_umap_delaunay":
            bases[name] = raw_delaunay_edges
        elif name == "learned_hd_knn":
            bases[name] = learned_hd.edge_index.to(device)
        elif name == "raw_hd_knn":
            bases[name] = raw_hd.edge_index.to(device)
        elif name == "delaunay_union_original":
            bases[name] = to_undirected(torch.cat((data.edge_index, delaunay_edges), dim=1), num_nodes=data.num_nodes)
        elif name != "delaunay" and name in controls:
            bases[name] = controls[name].edge_index.to(device)

    rewirer = VGAEDelaunayRewirer(add_self_loops=True, device=device)
    candidate_edges: dict[tuple[str, float], object] = {}
    rewiring_times: dict[tuple[str, float], float] = {}
    for constructor, base_edges in bases.items():
        ratios = pcfg.rewiring_ratios if constructor == "delaunay" or pcfg.refine_controls else (0.0,)
        for ratio in ratios:
            started = time.perf_counter()
            candidate_edges[(constructor, float(ratio))] = rewirer.rewire(
                base_edges, embeddings, float(ratio), int(data.num_nodes)
            )
            rewiring_times[(constructor, float(ratio))] = time.perf_counter() - started

    trainer = GNNTrainer(pipeline_config, device)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    baseline = trainer.train_single_run(
        data, pcfg.downstream_backbone, data.edge_index, gnn_params,
        seed=pipeline_seed, epochs=pcfg.downstream_epochs, track_test=True,
    )
    baseline_training_seconds = time.perf_counter() - started
    rows.append(_condition_row(synthetic, task, pipeline_seed, "original", "original", None, baseline, baseline_training_seconds, vgae_seconds, geometry_seconds, edge_index=data.edge_index, learned_trustworthiness=learned_trustworthiness, raw_trustworthiness=raw_trustworthiness))
    candidate_records=[]
    for (constructor, ratio), edges in candidate_edges.items():
        started = time.perf_counter()
        metrics = trainer.train_single_run(
            data, pcfg.downstream_backbone, edges, gnn_params,
            seed=pipeline_seed, epochs=pcfg.downstream_epochs, track_test=True,
        )
        training_seconds = time.perf_counter() - started
        row = _condition_row(
            synthetic, task, pipeline_seed,
            "geometry" if ratio == 0 else "refined",
            constructor, ratio, metrics, training_seconds,
            vgae_seconds, geometry_seconds, rewiring_times[(constructor, ratio)], edges,
            learned_trustworthiness, raw_trustworthiness,
        )
        rows.append(row); candidate_records.append(row)

    selection_key = f"val_{pcfg.selection_metric}"
    delaunay_records = [r for r in candidate_records if r["constructor"] == "delaunay"]
    selected = sorted(delaunay_records, key=lambda r: (-float(r[selection_key]), float(r["ratio"])))[0]
    geometry = next(r for r in delaunay_records if float(r["ratio"]) == 0.0)
    positive = [r for r in delaunay_records if float(r["ratio"]) > 0]
    selected_refinement = sorted(positive, key=lambda r: (-float(r[selection_key]), float(r["ratio"])))[0] if positive else geometry
    selected_row = {
        "run_id": f"{task.task_id}__p{pipeline_seed}",
        "graph_id": synthetic.graph_id,
        "family": synthetic.specification.family,
        "configuration_id": synthetic.specification.configuration_id,
        "task_id": task.task_id,
        "feature_regime": task.feature_regime,
        "task_seed": task.task_seed,
        "pipeline_seed": pipeline_seed,
        "baseline_test_f1": baseline["test_f1"],
        "geometry_test_f1": geometry["test_f1"],
        "selected_test_f1": selected["test_f1"],
        "refinement_test_f1": selected_refinement["test_f1"],
        "selected_ratio": selected["ratio"],
        "refinement_selected_ratio": selected_refinement["ratio"],
        "gain_geometry": float(geometry["test_f1"] - baseline["test_f1"]),
        "gain_vgae": float(selected_refinement["test_f1"] - geometry["test_f1"]),
        "gain_selected_rewiring": float(selected["test_f1"] - baseline["test_f1"]),
        "selection_metric": pcfg.selection_metric,
        "learned_umap_trustworthiness": learned_trustworthiness,
        "raw_umap_trustworthiness": raw_trustworthiness,
        **{f"param_{key}": value for key, value in synthetic.specification.parameters.items()},
    }
    delta_rows = _mechanism_rows(
        synthetic, task, pipeline_seed, data.num_nodes, data.edge_index,
        candidate_edges[("delaunay", 0.0)],
        candidate_edges[("delaunay", float(selected["ratio"]))], config,
    )
    return rows, selected_row, delta_rows


def _condition_row(
    synthetic, task, pipeline_seed, stage, constructor, ratio, metrics,
    training_seconds, vgae_seconds, geometry_seconds, rewiring_seconds=0.0,
    edge_index=None, learned_trustworthiness=float("nan"), raw_trustworthiness=float("nan"),
):
    return {
        "graph_id": synthetic.graph_id,
        "family": synthetic.specification.family,
        "configuration_id": synthetic.specification.configuration_id,
        "task_id": task.task_id,
        "feature_regime": task.feature_regime,
        "task_seed": task.task_seed,
        "pipeline_seed": pipeline_seed,
        "stage": stage,
        "constructor": constructor,
        "ratio": ratio,
        **metrics,
        "training_seconds": training_seconds,
        "vgae_seconds_shared": vgae_seconds,
        "geometry_seconds_shared": geometry_seconds,
        "rewiring_seconds": rewiring_seconds,
        "undirected_edges": int(edge_index.size(1) // 2) if edge_index is not None else None,
        "learned_umap_trustworthiness": learned_trustworthiness,
        "raw_umap_trustworthiness": raw_trustworthiness,
        **{f"param_{key}": value for key, value in synthetic.specification.parameters.items()},
    }


def _mechanism_rows(synthetic, task, pipeline_seed, num_nodes, original_edges, geometry_edges, selected_edges, config):
    from src.analysis.structural_metrics import edge_index_to_networkx
    graphs = {
        "original": edge_index_to_networkx(original_edges, int(num_nodes)),
        "delaunay": edge_index_to_networkx(geometry_edges, int(num_nodes)),
        "selected_refinement": edge_index_to_networkx(selected_edges, int(num_nodes)),
    }
    rows=[]
    for condition, graph in graphs.items():
        metrics=compute_prerewiring_metrics(graph, config.metrics, stable_seed(pipeline_seed, condition))
        rows.append({"graph_id": synthetic.graph_id, "task_id": task.task_id, "pipeline_seed": pipeline_seed, "condition": condition, **metrics})
    return rows


def aggregate_outcomes(config: SyntheticExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate seeds at graph level and configuration level."""
    selected = pd.read_csv(config.output_dir / "selected_execution_results.csv")
    metrics = pd.read_csv(config.output_dir / "synthetic_graph_metrics.csv")
    graph_outcomes = selected.groupby(["graph_id", "family", "configuration_id"], as_index=False).agg(
        gain_selected_rewiring=("gain_selected_rewiring", "mean"),
        gain_selected_rewiring_std=("gain_selected_rewiring", "std"),
        gain_geometry=("gain_geometry", "mean"),
        gain_vgae=("gain_vgae", "mean"),
        n_nested_runs=("gain_selected_rewiring", "size"),
        positive_run_fraction=("gain_selected_rewiring", lambda x: float(np.mean(np.asarray(x) > 0))),
    )
    for threshold in config.indicator.success_thresholds_pp:
        suffix = str(threshold).replace(".", "_")
        graph_outcomes[f"success_{suffix}"] = (100 * graph_outcomes["gain_selected_rewiring"] > threshold).astype(int)
        graph_outcomes[f"refine_success_{suffix}"] = (100 * graph_outcomes["gain_vgae"] > threshold).astype(int)
    graph_outcomes = metrics.merge(graph_outcomes, on=["graph_id", "family", "configuration_id"], how="inner")
    _atomic_csv(graph_outcomes, config.output_dir / "graph_level_outcomes.csv")
    numeric = ["gain_selected_rewiring", "gain_geometry", "gain_vgae", "positive_run_fraction"]
    config_outcomes = graph_outcomes.groupby(["family", "configuration_id"], as_index=False)[numeric].agg(["mean", "std", "count"])
    config_outcomes.columns = ["_".join(filter(None, map(str, col))) for col in config_outcomes.columns.to_flat_index()]
    _atomic_csv(config_outcomes, config.output_dir / "configuration_level_outcomes.csv")
    regime = selected.groupby(["graph_id", "feature_regime"], as_index=False).agg(
        gain_selected_rewiring=("gain_selected_rewiring", "mean"), gain_geometry=("gain_geometry", "mean"), gain_vgae=("gain_vgae", "mean"), n=("gain_vgae", "size")
    )
    _atomic_csv(regime, config.output_dir / "feature_regime_outcomes.csv")
    executions_path = config.output_dir / "all_execution_results.csv"
    if executions_path.exists():
        executions = pd.read_csv(executions_path)
        keys = ["graph_id", "task_id", "pipeline_seed"]
        original = executions[executions["constructor"].eq("original")][keys + ["test_f1"]].rename(
            columns={"test_f1": "original_test_f1"}
        )
        compared = executions.merge(original, on=keys, how="left")
        compared["gain_vs_original"] = compared["test_f1"] - compared["original_test_f1"]
        constructor_summary = compared.groupby(
            ["constructor", "ratio", "feature_regime"], dropna=False, as_index=False
        ).agg(
            mean_gain=("gain_vs_original", "mean"),
            median_gain=("gain_vs_original", "median"),
            mean_test_f1=("test_f1", "mean"),
            positive_fraction=("gain_vs_original", lambda x: float(np.mean(np.asarray(x) > 0))),
            n=("gain_vs_original", "size"),
        )
        _atomic_csv(constructor_summary, config.output_dir / "constructor_level_outcomes.csv")
    mechanism_path = config.output_dir / "postrewiring_mechanism_metrics.csv"
    if mechanism_path.exists():
        _postrewiring_deltas(pd.read_csv(mechanism_path), config.output_dir / "postrewiring_deltas.csv")
    return graph_outcomes, config_outcomes


def design_summary(config: SyntheticExperimentConfig) -> dict[str, int | str]:
    """Return and persist the exact factorial workload before execution."""
    from .generation import configuration_grid
    configurations = len(configuration_grid(config.graph))
    graphs = configurations * config.graph.instances_per_configuration
    tasks = graphs * len(config.task.feature_regimes) * config.task.task_seeds_per_graph
    nested_runs = tasks * config.pipeline.pipeline_seeds_per_task
    refined = len(config.pipeline.rewiring_ratios)
    controls = max(0, len(config.pipeline.constructors) - 1)
    conditions_per_run = 1 + refined + controls * (refined if config.pipeline.refine_controls else 1)
    summary = {
        "profile": config.profile,
        "configurations": configurations,
        "independent_graph_instances": graphs,
        "feature_label_tasks": tasks,
        "nested_pipeline_runs": nested_runs,
        "gnn_conditions_per_nested_run": conditions_per_run,
        "estimated_gnn_fits": nested_runs * conditions_per_run,
        "scientific_unit_for_indicator": "independent_graph_instance",
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _collect_parts(parts_dir: Path, pattern: str, output: Path) -> pd.DataFrame:
    files = sorted(parts_dir.glob(pattern))
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else pd.DataFrame()
    _atomic_csv(frame, output)
    return frame


def _postrewiring_deltas(frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    if frame.empty:
        result = pd.DataFrame(); _atomic_csv(result, output); return result
    keys = ["graph_id", "task_id", "pipeline_seed"]
    id_columns = set(keys + ["condition"])
    metrics = [c for c in frame.columns if c not in id_columns and pd.api.types.is_numeric_dtype(frame[c])]
    original = frame[frame["condition"].eq("original")][keys + metrics].copy()
    original = original.rename(columns={m: f"original_{m}" for m in metrics})
    rows=[]
    for condition in ("delaunay", "selected_refinement"):
        part=frame[frame["condition"].eq(condition)][keys+metrics].copy()
        merged=part.merge(original,on=keys,how="left")
        for metric in metrics:
            merged[f"delta_{metric}"]=merged[metric]-merged[f"original_{metric}"]
        merged["condition"]=condition
        rows.append(merged[keys+["condition"]+[f"delta_{m}" for m in metrics]])
    result=pd.concat(rows,ignore_index=True)
    _atomic_csv(result,output)
    return result


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _require_pipeline_dependencies() -> None:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        import umap  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The execution phase requires the project dependencies from requirements.txt. "
            "Graph generation and indicator analysis can run separately."
        ) from exc
