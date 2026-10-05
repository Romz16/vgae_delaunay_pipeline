#!/usr/bin/env python3
"""Run the auxiliary-only and disjoint-validation ablation on selected datasets."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DelaunayConfig, OptunaConfig, OutputConfig, PipelineConfig, RewiringConfig, SplitConfig, TrainingConfig
from src.data.loaders import GraphLoader
from src.data.splits import apply_nested_node_split
from src.models.gnn import GNNParams
from src.pipeline.orchestrator import VGAEDelaunayPipeline
from src.rewiring.delaunay_builder import DelaunayGraphBuilder
from src.training.gnn_trainer import GNNTrainer
from src.utils.device import resolve_device
from src.utils.logging_utils import configure_logging


DEFAULT_DATASETS = ("Airports-USA", "Cora", "Amazon-Photo", "Roman-Empire")


def slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def parse_overrides(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --source-override {value!r}; expected DATASET=PATH")
        dataset, path = value.split("=", 1)
        result[dataset.lower()] = Path(path)
    return result


def source_dir(dataset: str, root: Path, overrides: dict[str, Path]) -> Path:
    candidate = overrides.get(dataset.lower(), root / slug(dataset))
    if not (candidate / "pipeline_config.json").exists():
        raise FileNotFoundError(f"Missing completed source run for {dataset}: {candidate}")
    return candidate


def load_config(source: Path, device_name: str, auxiliary_epochs: int | None) -> PipelineConfig:
    payload = json.loads((source / "pipeline_config.json").read_text(encoding="utf-8"))
    output_payload = dict(payload["output"])
    output_payload["root_dir"] = source
    delaunay_payload = dict(payload["delaunay"])
    delaunay_payload.setdefault("trustworthiness_max_samples", 3000)
    if auxiliary_epochs is not None:
        delaunay_payload["pretrain_epochs"] = auxiliary_epochs
    return PipelineConfig(
        dataset_name=payload["dataset_name"],
        base_seed=int(payload["base_seed"]),
        run_seed_start=int(payload["run_seed_start"]),
        rewiring_ratios=tuple(payload["rewiring_ratios"]),
        backbones=tuple(payload["backbones"]),
        vgae_embedding_filename=payload.get("vgae_embedding_filename"),
        output_csv_filename=payload.get("output_csv_filename"),
        output_md_filename=payload.get("output_md_filename"),
        add_self_loops_after_rewire=bool(payload["add_self_loops_after_rewire"]),
        device=device_name,
        split=SplitConfig(**payload["split"]),
        rewiring=RewiringConfig(**payload["rewiring"]),
        optuna=OptunaConfig(**payload["optuna"]),
        training=TrainingConfig(**payload["training"]),
        delaunay=DelaunayConfig(**delaunay_payload),
        output=OutputConfig(**output_payload),
    )


def find_embeddings(source: Path, config: PipelineConfig) -> Path:
    filename = config.vgae_embedding_filename or f"{config.safe_dataset_name}_vgae_embeddings.npy"
    candidates = (source / filename, source / "embeddings" / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"VGAE embeddings not found under {source}: {filename}")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_one(
    dataset: str,
    seed: int,
    *,
    data_root: Path,
    source: Path,
    output_dir: Path,
    device_name: str,
    backbones: tuple[str, ...],
    epochs: int | None,
    auxiliary_epochs: int | None,
) -> None:
    import torch

    device = resolve_device(device_name)
    config = load_config(source, device_name, auxiliary_epochs)
    config = replace(config, dataset_name=dataset)
    data = GraphLoader(root=data_root, device=device).load_from_name(dataset)
    data = apply_nested_node_split(data, seed, device)
    mask_sizes = {
        name: int(getattr(data, f"{name}_mask").sum().item())
        for name in ("train", "inner_val", "topology_val", "test")
    }

    builder = DelaunayGraphBuilder(config, device)
    auxiliary_model, auxiliary = builder.fit_auxiliary_gcn(
        data,
        seed=seed,
        train_mask=data.train_mask,
        validation_mask=data.inner_val_mask,
        selection_metric=config.training.selection_metric,
    )
    delaunay_edges, _, _ = builder.build_from_auxiliary_model(data, auxiliary_model)
    embeddings = np.load(find_embeddings(source, config))
    if embeddings.shape[0] != int(data.num_nodes):
        raise ValueError(f"Embedding/node mismatch for {dataset}: {embeddings.shape[0]} != {data.num_nodes}")
    rewired_graphs = VGAEDelaunayPipeline(config, device)._build_rewired_graphs(
        delaunay_edges, embeddings, int(data.num_nodes)
    )

    params_payload = json.loads((source / "best_gnn_params.json").read_text(encoding="utf-8"))
    params_by_backbone = {name: GNNParams(**values) for name, values in params_payload.items()}
    trainer = GNNTrainer(config, device)
    wide_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    downstream_epochs = epochs or config.training.gnn_final_epochs

    for backbone in backbones:
        if backbone not in params_by_backbone:
            raise KeyError(f"No tuned parameters for backbone {backbone} in {source}")
        params = params_by_backbone[backbone]
        started = time.perf_counter()
        original = trainer.train_single_run_nested(
            data, backbone, data.edge_index, params, seed, downstream_epochs
        )
        original_seconds = time.perf_counter() - started

        candidates: list[dict[str, object]] = []
        for ratio, edge_index in sorted(rewired_graphs.items()):
            started = time.perf_counter()
            metrics = trainer.train_single_run_nested(
                data, backbone, edge_index, params, seed, downstream_epochs
            )
            row = {
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "candidate": f"delaunay_vgae_r{ratio:g}",
                "ratio": float(ratio),
                **metrics,
                "training_seconds": time.perf_counter() - started,
            }
            candidates.append(row)
            candidate_rows.append(row)
        metric = config.training.selection_metric.lower()
        selection_column = f"topology_val_{metric}"
        selected = sorted(candidates, key=lambda row: (-float(row[selection_column]), float(row["ratio"])))[0]
        wide_rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "split_id": f"nested-{seed}",
                **{f"n_{name}": count for name, count in mask_sizes.items()},
                "original_inner_val_f1": original["inner_val_f1"],
                "original_topology_val_f1": original["topology_val_f1"],
                "original_test_f1": original["test_f1"],
                "original_training_seconds": original_seconds,
                "auxiliary_inner_val_f1": auxiliary["inner_val_f1"],
                "auxiliary_topology_val_f1": auxiliary["topology_val_f1"],
                "auxiliary_test_f1": auxiliary["test_f1"],
                "auxiliary_best_epoch": auxiliary["best_epoch"],
                "auxiliary_training_seconds": auxiliary["elapsed_seconds"],
                "selected_candidate": selected["candidate"],
                "selected_ratio": selected["ratio"],
                "selected_inner_val_f1": selected["inner_val_f1"],
                "selected_topology_val_f1": selected["topology_val_f1"],
                "selected_test_f1": selected["test_f1"],
                "selected_training_seconds": selected["training_seconds"],
                "gain_selected_vs_original": selected["test_f1"] - original["test_f1"],
                "gain_auxiliary_vs_original": auxiliary["test_f1"] - original["test_f1"],
                "gain_selected_vs_auxiliary": selected["test_f1"] - auxiliary["test_f1"],
                "checkpoint_selection_split": "inner_validation",
                "topology_selection_split": "topology_validation",
                "final_evaluation_split": "test",
            }
        )

    run_id = f"{slug(dataset)}__{seed}"
    atomic_csv(pd.DataFrame(wide_rows), output_dir / "run_parts" / f"{run_id}.selected.csv")
    atomic_csv(pd.DataFrame(candidate_rows), output_dir / "run_parts" / f"{run_id}.candidates.csv")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = {
        "selected_vs_original": "gain_selected_vs_original",
        "auxiliary_vs_original": "gain_auxiliary_vs_original",
        "selected_vs_auxiliary": "gain_selected_vs_auxiliary",
    }
    for (dataset, backbone), group in selected.groupby(["dataset", "backbone"], sort=True):
        for comparison, column in comparisons.items():
            values = group[column].to_numpy(dtype=float)
            try:
                statistic, p_value = wilcoxon(values, alternative="two-sided")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            rng = np.random.default_rng(20260920)
            bootstrap = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "comparison": comparison,
                    "n": len(values),
                    "mean_gain_pp": 100.0 * float(np.mean(values)),
                    "median_gain_pp": 100.0 * float(np.median(values)),
                    "std_gain_pp": 100.0 * float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "bootstrap_ci95_low_pp": 100.0 * float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci95_high_pp": 100.0 * float(np.quantile(bootstrap, 0.975)),
                    "positive_runs_pct": 100.0 * float(np.mean(values > 0.0)),
                    "wilcoxon_statistic": float(statistic),
                    "wilcoxon_p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def aggregate(output_dir: Path) -> None:
    selected_paths = sorted((output_dir / "run_parts").glob("*.selected.csv"))
    candidate_paths = sorted((output_dir / "run_parts").glob("*.candidates.csv"))
    if not selected_paths:
        raise FileNotFoundError(f"No completed run parts in {output_dir / 'run_parts'}")
    selected = pd.concat((pd.read_csv(path) for path in selected_paths), ignore_index=True)
    candidates = pd.concat((pd.read_csv(path) for path in candidate_paths), ignore_index=True)
    selected = selected.sort_values(["dataset", "backbone", "seed"])
    candidates = candidates.sort_values(["dataset", "backbone", "seed", "ratio"])
    atomic_csv(selected, output_dir / "nested_auxiliary_selected_per_seed.csv")
    atomic_csv(candidates, output_dir / "nested_auxiliary_candidates_long.csv")
    atomic_csv(summarize(selected), output_dir / "nested_auxiliary_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-override", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--backbones", nargs="+", default=["gcn", "sage", "gat"])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=12345)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--auxiliary-epochs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(output_dir)
        return
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    root = Path(args.source_root)
    overrides = parse_overrides(args.source_override)
    work = [(dataset, args.seed_start + offset) for dataset in args.datasets for offset in range(args.runs)]
    for index, (dataset, seed) in enumerate(work):
        if index % args.num_shards != args.shard_index:
            continue
        part = output_dir / "run_parts" / f"{slug(dataset)}__{seed}.selected.csv"
        candidate_part = output_dir / "run_parts" / f"{slug(dataset)}__{seed}.candidates.csv"
        if part.exists() and candidate_part.exists():
            continue
        run_one(
            dataset,
            seed,
            data_root=Path(args.data_root),
            source=source_dir(dataset, root, overrides),
            output_dir=output_dir,
            device_name=args.device,
            backbones=tuple(args.backbones),
            epochs=args.epochs,
            auxiliary_epochs=args.auxiliary_epochs,
        )
    if args.finalize:
        aggregate(output_dir)


if __name__ == "__main__":
    main()
