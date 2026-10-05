#!/usr/bin/env python3

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development hosts
    fcntl = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PipelineConfig
from scripts.run_raw_feature_ablation import load_pipeline_config
from src.data.loaders import GraphLoader
from src.data.splits import apply_node_split
from src.models.gnn import GNNParams
from src.pipeline.orchestrator import VGAEDelaunayPipeline
from src.rewiring.constructor_controls import build_constructor_controls
from src.rewiring.delaunay_builder import DelaunayGraphBuilder
from src.rewiring.hybrid_vgae_ct import HybridVGAECTDelaunayRewirer
from src.rewiring.sota_baselines import diffwire_ct_edge_weights, sdrf_snapshots
from src.rewiring.vgae_delaunay import VGAEDelaunayRewirer
from src.training.gnn_trainer import GNNTrainer
from src.utils.device import resolve_device
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_seed


DATASETS = [
    "Cora", "Coauthor-CS", "Pubmed", "Airports-USA", "Airports-Brazil", "Airports-Europe",
    "Amazon-Photo", "Cornell", "Actor", "Roman-Empire", "Texas", "Minesweeper",
]
PRIMARY_CONSTRUCTORS = (
    "knn",
    "mutual_knn",
    "random_edge_budget_matched",
    "random_degree_matched",
)
EXTENDED_CONSTRUCTORS = ("radius", "mst_knn")
REPRESENTATIVE_DATASETS = frozenset({"Cora", "Pubmed", "Actor", "Roman-Empire"})
OFFICIAL_REFS = {
    "SDRF": {"repository": "https://github.com/jctops/understanding-oversquashing", "commit": "c14e91d31ec5e9ff2dc13d90ef61b9dc2bdce829"},
    "DiffWire": {"repository": "https://github.com/AdrianArnaiz/DiffWire", "commit": "deced4bbe088827e39a9359fa368a8efa2b00cfd"},
}


def slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def constructors_for_dataset(dataset: str) -> tuple[str, ...]:
    if dataset in REPRESENTATIVE_DATASETS:
        return PRIMARY_CONSTRUCTORS + EXTENDED_CONSTRUCTORS
    return PRIMARY_CONSTRUCTORS


@contextmanager
def dataset_phase_lock(output_root: Path, dataset: str, phase: str):
    lock_dir = output_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{slug(dataset)}_{phase}.lock").open("w", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def source_output(dataset: str, cora_output: Path, protocol_root: Path) -> Path:
    return cora_output if dataset.lower() == "cora" else protocol_root / slug(dataset)


def load_params(output: Path) -> dict[str, GNNParams]:
    payload = json.loads((output / "best_gnn_params.json").read_text(encoding="utf-8"))
    return {name: GNNParams(**values) for name, values in payload.items()}


def baseline_lookup(output: Path) -> dict[tuple[str, int], dict[str, float]]:
    frame = pd.read_csv(output / "outputs_revision" / "selected_rates.csv")
    result = {}
    for _, row in frame.iterrows():
        result[(str(row["backbone"]), int(row["seed"]))] = {
            "baseline_val_acc": float(row["baseline_val_acc"]),
            "baseline_val_f1": float(row["baseline_val_f1"]),
            "baseline_test_acc": float(row["baseline_test_acc"]),
            "baseline_test_f1": float(row["baseline_test_f1"]),
        }
    return result


def rewirer(config: PipelineConfig, device):
    if config.rewiring.strategy.lower() == "hybrid_ct":
        return HybridVGAECTDelaunayRewirer(config.rewiring, config.add_self_loops_after_rewire, device)
    return VGAEDelaunayRewirer(config.add_self_loops_after_rewire, device)


def summary(selected: pd.DataFrame, method_col: str) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    return selected.groupby(["dataset", method_col, "backbone"], as_index=False).agg(
        n=("seed", "count"),
        selected_test_f1_mean=("selected_test_f1", "mean"),
        selected_test_f1_std=("selected_test_f1", "std"),
        selected_test_acc_mean=("selected_test_acc", "mean"),
        selected_test_acc_std=("selected_test_acc", "std"),
        gain_f1_mean=("gain_f1", "mean"),
        gain_f1_std=("gain_f1", "std"),
        gain_acc_mean=("gain_acc", "mean"),
        gain_acc_std=("gain_acc", "std"),
    )


def evaluate_candidates(
    trainer: GNNTrainer,
    data,
    dataset: str,
    method: str,
    backbone: str,
    params: GNNParams,
    candidates: dict[float, object],
    baseline: dict[tuple[str, int], dict[str, float]],
    final_runs: int,
    epochs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows, selected_rows = [], []
    selection_col = f"val_{trainer.config.training.selection_metric.lower()}"
    for run_idx in range(final_runs):
        seed = trainer.config.run_seed_start + run_idx
        rows = []
        for ratio, edge_index in sorted(candidates.items()):
            metrics = trainer.train_single_run(data, backbone, edge_index, params, seed, epochs, track_test=True)
            row = {"dataset": dataset, "method": method, "backbone": backbone, "seed": seed, "candidate_ratio": ratio, **metrics}
            rows.append(row)
            all_rows.append(row)
        chosen = sorted(rows, key=lambda row: (-float(row[selection_col]), float(row["candidate_ratio"])))[0]
        base = baseline[(backbone, seed)]
        selected_rows.append({
            "dataset": dataset, "method": method, "backbone": backbone, "seed": seed,
            "selected_ratio": chosen["candidate_ratio"], "selected_val_acc": chosen["val_acc"],
            "selected_val_f1": chosen["val_f1"], "selected_test_acc": chosen["test_acc"],
            "selected_test_f1": chosen["test_f1"], **base,
            "gain_acc": float(chosen["test_acc"]) - base["baseline_test_acc"],
            "gain_f1": float(chosen["test_f1"]) - base["baseline_test_f1"],
        })
    return all_rows, selected_rows


def run_sota(dataset: str, data, config: PipelineConfig, params, baseline, output: Path, final_runs: int, device) -> None:
    output.mkdir(parents=True, exist_ok=True)
    trainer = GNNTrainer(config, device)
    all_path = output / "sota_all_runs.partial.csv"
    selected_path = output / "sota_selected_runs.partial.csv"
    all_rows = pd.read_csv(all_path).to_dict("records") if all_path.exists() else []
    selected_rows = pd.read_csv(selected_path).to_dict("records") if selected_path.exists() else []
    sdrf_graphs = sdrf_snapshots(
        data.edge_index, int(data.num_nodes), tuple(config.rewiring_ratios), seed=config.base_seed,
        temperature=5.0, removal_bound=None,
    )
    for backbone in config.backbones:
        completed = sum(1 for row in selected_rows if row["method"] == "SDRF" and row["backbone"] == backbone)
        if completed == final_runs:
            continue
        all_rows = [row for row in all_rows if not (row["method"] == "SDRF" and row["backbone"] == backbone)]
        selected_rows = [row for row in selected_rows if not (row["method"] == "SDRF" and row["backbone"] == backbone)]
        rows, selected = evaluate_candidates(
            trainer, data, dataset, "SDRF", backbone, params[backbone], sdrf_graphs,
            baseline, final_runs, config.training.gnn_final_epochs,
        )
        all_rows.extend(rows)
        selected_rows.extend(selected)
        pd.DataFrame(all_rows).to_csv(all_path, index=False)
        pd.DataFrame(selected_rows).to_csv(selected_path, index=False)

    ct_weights, ct_meta = diffwire_ct_edge_weights(data.edge_index, int(data.num_nodes), config.rewiring)
    ct_completed = sum(1 for row in selected_rows if row["method"] == "DiffWire CT" and row["backbone"] == "gcn")
    if ct_completed != final_runs:
        all_rows = [row for row in all_rows if row["method"] != "DiffWire CT"]
        selected_rows = [row for row in selected_rows if row["method"] != "DiffWire CT"]
        for run_idx in range(final_runs):
            seed = config.run_seed_start + run_idx
            metrics = trainer.train_single_run(
                data, "gcn", data.edge_index, params["gcn"], seed, config.training.gnn_final_epochs,
                track_test=True, edge_weight=ct_weights,
            )
            row = {"dataset": dataset, "method": "DiffWire CT", "backbone": "gcn", "seed": seed, "candidate_ratio": np.nan, **metrics}
            all_rows.append(row)
            base = baseline[("gcn", seed)]
            selected_rows.append({
                "dataset": dataset, "method": "DiffWire CT", "backbone": "gcn", "seed": seed,
                "selected_ratio": np.nan, "selected_val_acc": metrics["val_acc"], "selected_val_f1": metrics["val_f1"],
                "selected_test_acc": metrics["test_acc"], "selected_test_f1": metrics["test_f1"], **base,
                "gain_acc": float(metrics["test_acc"]) - base["baseline_test_acc"],
                "gain_f1": float(metrics["test_f1"]) - base["baseline_test_f1"],
            })
        pd.DataFrame(all_rows).to_csv(all_path, index=False)
        pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    all_frame, selected_frame = pd.DataFrame(all_rows), pd.DataFrame(selected_rows)
    all_frame.to_csv(output / "sota_all_runs.csv", index=False)
    selected_frame.to_csv(output / "sota_selected_runs.csv", index=False)
    summary(selected_frame, "method").to_csv(output / "sota_summary.csv", index=False)
    (output / "sota_protocol_metadata.json").write_text(json.dumps({
        "dataset": dataset, "official_references": OFFICIAL_REFS, "diffwire_ct": ct_meta,
        "split": config.split.__dict__, "base_seed": config.base_seed, "run_seed_start": config.run_seed_start,
        "final_runs": final_runs, "gnn_trials_reused": config.optuna.gnn_trials,
        "gnn_opt_epochs": config.training.gnn_opt_epochs, "gnn_final_epochs": config.training.gnn_final_epochs,
        "selection_metric": config.training.selection_metric, "sdrf_temperature": 5.0,
        "sdrf_step_ratios_times_num_nodes": list(config.rewiring_ratios), "sdrf_removal_bound": None,
    }, indent=2), encoding="utf-8")


def run_controls(dataset: str, data, config: PipelineConfig, params, baseline, source: Path, output: Path, final_runs: int, device) -> None:
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.base_seed)
    seeded_data = apply_node_split(data, config.base_seed, config.split, device)
    builder = DelaunayGraphBuilder(config, device)
    delaunay_edge_index, learned_features, points = builder.build_with_projection(seeded_data)
    protocol_metrics = pd.read_csv(source / "outputs_revision" / "selected_rewired_graph_metrics.csv")
    zero_rows = protocol_metrics[protocol_metrics["graph_name"].astype(str).str.endswith("0%")]
    if zero_rows.empty:
        raise RuntimeError(f"Could not locate the original protocol's Delaunay 0% metric for {dataset}.")
    expected_delaunay_edges = int(zero_rows.iloc[0]["num_edges"])
    actual_delaunay_edges = int(delaunay_edge_index.size(1) // 2)
    edge_count_difference = abs(actual_delaunay_edges - expected_delaunay_edges)
    reproduction_tolerance = max(10, int(np.ceil(expected_delaunay_edges * 0.001)))
    if edge_count_difference > reproduction_tolerance:
        raise RuntimeError(
            f"Auxiliary-GCN/Delaunay reproduction mismatch for {dataset}: "
            f"current={actual_delaunay_edges}, original_protocol={expected_delaunay_edges}, "
            f"difference={edge_count_difference}, tolerance={reproduction_tolerance}."
        )
    if edge_count_difference:
        logging.warning(
            "Accepted numerical Delaunay edge-count variation for %s: current=%d, original=%d, "
            "difference=%d, tolerance=%d",
            dataset, actual_delaunay_edges, expected_delaunay_edges,
            edge_count_difference, reproduction_tolerance,
        )
    all_controls = build_constructor_controls(points, delaunay_edge_index, seed=config.base_seed)
    requested_controls = constructors_for_dataset(dataset)
    controls = {name: all_controls[name] for name in requested_controls}
    embeddings = np.load(source / "embeddings" / f"{config.safe_dataset_name}_vgae_embeddings.npy")
    method = rewirer(config, device)
    metadata = {
        "dataset": dataset,
        "shared_representation": "auxiliary GCN first hidden layer -> identical UMAP 2-D coordinates",
        "delaunay_undirected_edges": actual_delaunay_edges,
        "original_protocol_delaunay_edges": expected_delaunay_edges,
        "exact_representation_reproduction_check": actual_delaunay_edges == expected_delaunay_edges,
        "edge_count_difference": edge_count_difference,
        "reproduction_tolerance_edges": reproduction_tolerance,
        "reproduction_within_tolerance": True,
        "control_profile": "lean_30_seed",
        "primary_constructors_all_datasets": list(PRIMARY_CONSTRUCTORS),
        "extended_constructors_representative_datasets": list(EXTENDED_CONSTRUCTORS),
        "representative_datasets": sorted(REPRESENTATIVE_DATASETS),
        "constructors": {},
    }
    all_path = output / "constructor_all_runs.partial.csv"
    selected_path = output / "constructor_selected_runs.partial.csv"
    all_rows = pd.read_csv(all_path).to_dict("records") if all_path.exists() else []
    selected_rows = pd.read_csv(selected_path).to_dict("records") if selected_path.exists() else []
    for name, control in controls.items():
        metadata["constructors"][name] = control.metadata
        candidate_graphs = {
            ratio: method.rewire(control.edge_index, embeddings, ratio, int(data.num_nodes))
            for ratio in config.rewiring_ratios
        }
        for backbone in config.backbones:
            completed = sum(1 for row in selected_rows if row["method"] == name and row["backbone"] == backbone)
            if completed == final_runs:
                continue
            all_rows = [row for row in all_rows if not (row["method"] == name and row["backbone"] == backbone)]
            selected_rows = [row for row in selected_rows if not (row["method"] == name and row["backbone"] == backbone)]
            rows, selected = evaluate_candidates(
                GNNTrainer(config, device), data, dataset, name, backbone, params[backbone], candidate_graphs,
                baseline, final_runs, config.training.gnn_final_epochs,
            )
            all_rows.extend(rows)
            selected_rows.extend(selected)
            pd.DataFrame(all_rows).to_csv(all_path, index=False)
            pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    all_frame, selected_frame = pd.DataFrame(all_rows), pd.DataFrame(selected_rows)
    all_frame.to_csv(output / "constructor_all_runs.csv", index=False)
    selected_frame.to_csv(output / "constructor_selected_runs.csv", index=False)
    summary(selected_frame, "method").to_csv(output / "constructor_summary.csv", index=False)
    metadata.update({"rewiring_strategy": config.rewiring.strategy, "rewiring_ratios": list(config.rewiring_ratios), "backbones": list(config.backbones), "final_runs": final_runs, "selection_metric": config.training.selection_metric})
    (output / "constructor_protocol_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cora-output", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase", choices=["sota", "constructors", "all"], default="all")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--final-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    for dataset in args.datasets:
        phases = ("sota", "constructors") if args.phase == "all" else (args.phase,)
        with ExitStack() as locks:
            for phase in phases:
                locks.enter_context(dataset_phase_lock(output_root, dataset, phase))
            source = source_output(dataset, Path(args.cora_output), Path(args.protocol_root))
            config = load_pipeline_config(source, args.device)
            final_runs = args.final_runs or config.training.final_runs
            data = GraphLoader(args.data_root, device).load_from_name(dataset)
            params = load_params(source)
            baseline = baseline_lookup(source)
            dataset_output = output_root / slug(dataset)
            if args.phase in {"sota", "all"}:
                run_sota(dataset, data, config, params, baseline, dataset_output / "sota", final_runs, device)
            if args.phase in {"constructors", "all"}:
                run_controls(dataset, data, config, params, baseline, source, dataset_output / "constructors", final_runs, device)


if __name__ == "__main__":
    main()
