#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DelaunayConfig, OptunaConfig, OutputConfig, PipelineConfig, RewiringConfig, SplitConfig, TrainingConfig
from src.data.graph_corruption import corrupt_graph
from src.data.loaders import GraphLoader
from src.data.splits import apply_nested_node_split
from src.models.gnn import GNNParams
from src.models.vgae import VGAEParams
from src.rewiring.delaunay_builder import DelaunayGraphBuilder
from src.rewiring.proximity_graphs import build_proximity_graphs
from src.rewiring.vgae_delaunay import VGAEDelaunayRewirer
from src.training.gnn_trainer import GNNTrainer
from src.training.vgae_trainer import VGAETrainer
from src.utils.device import resolve_device
from src.utils.logging_utils import configure_logging


DEFAULT_DATASETS = ("Airports-USA", "Cora", "Amazon-Photo", "Roman-Empire")
DEFAULT_CONSTRUCTORS = ("delaunay", "gabriel", "rng", "mst")
DEFAULT_RATIOS = (0.0, 0.25, 0.55)
DEFAULT_CORRUPTIONS = ("none:0", "degree_preserving:0.25", "homophily_attack:0.25")
REPRESENTATION_ENCODERS = ("aux_gcn", "vgae")
REFINEMENT_ENCODERS = ("aux_gcn", "vgae")


def slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_").replace(".", "p")


def stable_seed(*parts: object) -> int:
    payload = "::".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") % (2**31 - 1)


def parse_overrides(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid source override {value!r}; expected DATASET=PATH")
        dataset, path = value.split("=", 1)
        result[dataset.lower()] = Path(path)
    return result


def source_dir(dataset: str, root: Path, overrides: dict[str, Path]) -> Path:
    candidate = overrides.get(dataset.lower(), root / slug(dataset))
    if not (candidate / "pipeline_config.json").exists():
        raise FileNotFoundError(f"Missing completed source run for {dataset}: {candidate}")
    return candidate


def parse_corruptions(values: list[str]) -> list[tuple[str, float]]:
    parsed: list[tuple[str, float]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid corruption {value!r}; expected MODE:FRACTION")
        mode, fraction = value.split(":", 1)
        parsed.append((mode.lower(), float(fraction)))
    return parsed


def load_config(source: Path, device_name: str, auxiliary_epochs: int, ratios: tuple[float, ...]) -> PipelineConfig:
    payload = json.loads((source / "pipeline_config.json").read_text(encoding="utf-8"))
    output_payload = dict(payload["output"]); output_payload["root_dir"] = source
    delaunay_payload = dict(payload["delaunay"])
    delaunay_payload["pretrain_epochs"] = auxiliary_epochs
    delaunay_payload.setdefault("trustworthiness_max_samples", 3000)
    return PipelineConfig(
        dataset_name=payload["dataset_name"],
        base_seed=int(payload["base_seed"]),
        run_seed_start=int(payload["run_seed_start"]),
        rewiring_ratios=ratios,
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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_params(source: Path) -> tuple[dict[str, GNNParams], VGAEParams]:
    gnn_payload = json.loads((source / "best_gnn_params.json").read_text(encoding="utf-8"))
    vgae_payload = json.loads((source / "best_vgae_params.json").read_text(encoding="utf-8"))
    return ({name: GNNParams(**values) for name, values in gnn_payload.items()}, VGAEParams(**vgae_payload))


def embeddings_for_observed_graph(
    data,
    observed_edges: torch.Tensor,
    *,
    config: PipelineConfig,
    vgae_params: VGAEParams,
    seed: int,
    vgae_epochs: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    observed_data = data.clone()
    observed_data.edge_index = observed_edges
    builder = DelaunayGraphBuilder(config, observed_edges.device)

    started = time.perf_counter()
    auxiliary_model, auxiliary_metrics = builder.fit_auxiliary_gcn(
        observed_data,
        seed=seed,
        train_mask=observed_data.train_mask,
        validation_mask=observed_data.inner_val_mask,
        selection_metric=config.training.selection_metric,
    )
    with torch.no_grad():
        auxiliary_embeddings = auxiliary_model.get_embeddings(observed_data.x, observed_edges).detach().cpu().numpy().astype(np.float32)
    auxiliary_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vgae_trainer = VGAETrainer(observed_edges.device)
    vgae_model = vgae_trainer.train(
        observed_data,
        vgae_params,
        epochs=vgae_epochs,
        seed=stable_seed(seed, "vgae"),
        edge_index=observed_edges,
    )
    vgae_embeddings = vgae_trainer.extract_embeddings(vgae_model, observed_data, observed_edges)
    vgae_seconds = time.perf_counter() - started
    return (
        {"aux_gcn": auxiliary_embeddings, "vgae": vgae_embeddings},
        {
            "auxiliary_training_seconds": auxiliary_seconds,
            "vgae_training_seconds": vgae_seconds,
            "auxiliary_inner_val_f1": float(auxiliary_metrics["inner_val_f1"]),
            "auxiliary_topology_val_f1": float(auxiliary_metrics["topology_val_f1"]),
            "auxiliary_test_f1": float(auxiliary_metrics["test_f1"]),
        },
    )


def run_one(
    dataset: str,
    seed: int,
    corruption_mode: str,
    corruption_fraction: float,
    *,
    data_root: Path,
    source: Path,
    output_dir: Path,
    device_name: str,
    backbones: tuple[str, ...],
    constructors: tuple[str, ...],
    ratios: tuple[float, ...],
    downstream_epochs: int,
    auxiliary_epochs: int,
    vgae_epochs: int,
    gate_margin_pp: float,
) -> None:
    device = resolve_device(device_name)
    config = replace(load_config(source, device_name, auxiliary_epochs, ratios), dataset_name=dataset)
    data = GraphLoader(root=data_root, device=device).load_from_name(dataset)
    data = apply_nested_node_split(data, seed, device)
    clean_edges = data.edge_index
    corruption = corrupt_graph(
        clean_edges,
        data.y,
        num_nodes=int(data.num_nodes),
        mode=corruption_mode,
        fraction=corruption_fraction,
        seed=stable_seed(dataset, seed, corruption_mode, corruption_fraction),
        device=device,
    )
    observed_edges = corruption.edge_index
    embeddings, encoder_metrics = embeddings_for_observed_graph(
        data,
        observed_edges,
        config=config,
        vgae_params=load_params(source)[1],
        seed=seed,
        vgae_epochs=vgae_epochs,
    )
    gnn_params, _ = load_params(source)
    builder = DelaunayGraphBuilder(config, device)
    projections: dict[str, np.ndarray] = {}
    graph_sets: dict[str, dict[str, object]] = {}
    for representation_encoder in REPRESENTATION_ENCODERS:
        points = builder._project_umap(embeddings[representation_encoder])
        projections[representation_encoder] = points
        graph_sets[representation_encoder] = build_proximity_graphs(
            points,
            constructors=constructors,
            device=device,
        )

    trainer = GNNTrainer(config, device)
    rewirer = VGAEDelaunayRewirer(
        add_self_loops=config.add_self_loops_after_rewire,
        device=device,
    )
    candidates: list[dict[str, object]] = []
    method_selected: list[dict[str, object]] = []
    global_selected: list[dict[str, object]] = []
    corruption_id = f"{slug(corruption_mode)}_{slug(str(corruption_fraction))}"

    for backbone in backbones:
        if backbone not in gnn_params:
            raise KeyError(f"No tuned parameters for {backbone} under {source}")
        params = gnn_params[backbone]
        clean_metrics = trainer.train_single_run_nested(data, backbone, clean_edges, params, seed, downstream_epochs)
        observed_metrics = trainer.train_single_run_nested(data, backbone, observed_edges, params, seed, downstream_epochs)
        original_row = {
            "dataset": dataset,
            "backbone": backbone,
            "seed": seed,
            "corruption_mode": corruption_mode,
            "corruption_fraction": corruption_fraction,
            "representation_encoder": "original",
            "refinement_encoder": "original",
            "constructor": "original",
            "ratio": np.nan,
            "candidate": "observed_original",
            "undirected_edges": int(corruption.metadata["observed_edges"]),
            **encoder_metrics,
            **corruption.metadata,
            **observed_metrics,
        }
        candidates.append(original_row)

        for representation_encoder in REPRESENTATION_ENCODERS:
            for refinement_encoder in REFINEMENT_ENCODERS:
                for constructor in constructors:
                    base_graph = graph_sets[representation_encoder][constructor]
                    method_rows: list[dict[str, object]] = []
                    for ratio in ratios:
                        candidate_edges = rewirer.rewire(
                            base_graph.edge_index,
                            embeddings[refinement_encoder],
                            ratio,
                            int(data.num_nodes),
                        )
                        started = time.perf_counter()
                        metrics = trainer.train_single_run_nested(
                            data,
                            backbone,
                            candidate_edges,
                            params,
                            seed,
                            downstream_epochs,
                        )
                        row = {
                            "dataset": dataset,
                            "backbone": backbone,
                            "seed": seed,
                            "corruption_mode": corruption_mode,
                            "corruption_fraction": corruption_fraction,
                            "representation_encoder": representation_encoder,
                            "refinement_encoder": refinement_encoder,
                            "constructor": constructor,
                            "ratio": ratio,
                            "candidate": f"{representation_encoder}__{constructor}__{refinement_encoder}__r{ratio:g}",
                            "undirected_edges": int(base_graph.metadata["undirected_edges"]),
                            "downstream_training_seconds": time.perf_counter() - started,
                            **encoder_metrics,
                            **corruption.metadata,
                            **metrics,
                        }
                        candidates.append(row)
                        method_rows.append(row)
                    selected = sorted(
                        method_rows,
                        key=lambda row: (-float(row["topology_val_f1"]), float(row["ratio"])),
                    )[0]
                    method_selected.append(
                        {
                            **selected,
                            "clean_original_test_f1": clean_metrics["test_f1"],
                            "observed_original_test_f1": observed_metrics["test_f1"],
                            "gain_vs_observed_original": selected["test_f1"] - observed_metrics["test_f1"],
                            "gain_vs_clean_original": selected["test_f1"] - clean_metrics["test_f1"],
                            "corruption_damage": observed_metrics["test_f1"] - clean_metrics["test_f1"],
                        }
                    )

        eligible = [row for row in method_selected if row["dataset"] == dataset and row["backbone"] == backbone and row["seed"] == seed and row["corruption_mode"] == corruption_mode and float(row["corruption_fraction"]) == corruption_fraction]
        best = sorted(eligible, key=lambda row: -float(row["topology_val_f1"]))[0]
        advantage_pp = 100.0 * (float(best["topology_val_f1"]) - float(observed_metrics["topology_val_f1"]))
        use_rewiring = advantage_pp >= gate_margin_pp
        chosen = best if use_rewiring else original_row
        global_selected.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "seed": seed,
                "corruption_mode": corruption_mode,
                "corruption_fraction": corruption_fraction,
                "selected_candidate": chosen["candidate"],
                "selected_representation_encoder": chosen["representation_encoder"],
                "selected_refinement_encoder": chosen["refinement_encoder"],
                "selected_constructor": chosen["constructor"],
                "selected_ratio": chosen["ratio"],
                "gate_margin_pp": gate_margin_pp,
                "validation_advantage_pp": advantage_pp,
                "gate_used_rewiring": use_rewiring,
                "clean_original_test_f1": clean_metrics["test_f1"],
                "observed_original_test_f1": observed_metrics["test_f1"],
                "selected_test_f1": chosen["test_f1"],
                "gain_vs_observed_original": chosen["test_f1"] - observed_metrics["test_f1"],
                "gain_vs_clean_original": chosen["test_f1"] - clean_metrics["test_f1"],
                "corruption_damage": observed_metrics["test_f1"] - clean_metrics["test_f1"],
                **corruption.metadata,
            }
        )

    prefix = f"{slug(dataset)}__{seed}__{corruption_id}"
    atomic_csv(pd.DataFrame(candidates), output_dir / "run_parts" / f"{prefix}.candidates.csv")
    atomic_csv(pd.DataFrame(method_selected), output_dir / "run_parts" / f"{prefix}.methods.csv")
    atomic_csv(pd.DataFrame(global_selected), output_dir / "run_parts" / f"{prefix}.global.csv")
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _summary(frame: pd.DataFrame, group_columns: list[str], gain_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260924)
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        values = group[gain_column].to_numpy(dtype=float)
        bootstrap = np.mean(rng.choice(values, size=(5000, len(values)), replace=True), axis=1)
        try:
            statistic, p_value = wilcoxon(values, alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row.update(
            n=len(values),
            mean_gain_pp=100.0 * float(np.mean(values)),
            median_gain_pp=100.0 * float(np.median(values)),
            std_gain_pp=100.0 * float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            bootstrap_ci95_low_pp=100.0 * float(np.quantile(bootstrap, 0.025)),
            bootstrap_ci95_high_pp=100.0 * float(np.quantile(bootstrap, 0.975)),
            positive_runs_pct=100.0 * float(np.mean(values > 0.0)),
            wilcoxon_statistic=float(statistic),
            wilcoxon_p_value=float(p_value),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(output_dir: Path) -> None:
    parts = output_dir / "run_parts"
    kinds = {kind: sorted(parts.glob(f"*.{kind}.csv")) for kind in ("candidates", "methods", "global")}
    if any(not paths for paths in kinds.values()):
        raise FileNotFoundError(f"Incomplete run parts under {parts}")
    candidates = pd.concat((pd.read_csv(path) for path in kinds["candidates"]), ignore_index=True)
    methods = pd.concat((pd.read_csv(path) for path in kinds["methods"]), ignore_index=True)
    global_rows = pd.concat((pd.read_csv(path) for path in kinds["global"]), ignore_index=True)
    atomic_csv(candidates, output_dir / "candidate_runs_long.csv")
    atomic_csv(methods, output_dir / "method_selected_per_seed.csv")
    atomic_csv(global_rows, output_dir / "global_gate_selected_per_seed.csv")
    method_groups = ["dataset", "backbone", "corruption_mode", "corruption_fraction", "representation_encoder", "refinement_encoder", "constructor"]
    atomic_csv(_summary(methods, method_groups, "gain_vs_observed_original"), output_dir / "summary_by_method.csv")
    global_groups = ["dataset", "backbone", "corruption_mode", "corruption_fraction"]
    atomic_csv(_summary(global_rows, global_groups, "gain_vs_observed_original"), output_dir / "global_gate_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-override", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--backbones", nargs="+", default=["gcn", "sage", "gat"])
    parser.add_argument("--constructors", nargs="+", default=list(DEFAULT_CONSTRUCTORS))
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--corruptions", nargs="+", default=list(DEFAULT_CORRUPTIONS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=52000)
    parser.add_argument("--downstream-epochs", type=int, default=200)
    parser.add_argument("--auxiliary-epochs", type=int, default=150)
    parser.add_argument("--vgae-epochs", type=int, default=200)
    parser.add_argument("--gate-margin-pp", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(output_dir); return
    corruptions = parse_corruptions(args.corruptions)
    overrides = parse_overrides(args.source_override)
    root = Path(args.source_root)
    work = [
        (dataset, args.seed_start + run, mode, fraction)
        for dataset in args.datasets
        for run in range(args.runs)
        for mode, fraction in corruptions
    ]
    for index, (dataset, seed, mode, fraction) in enumerate(work):
        if index % args.num_shards != args.shard_index:
            continue
        prefix = f"{slug(dataset)}__{seed}__{slug(mode)}_{slug(str(fraction))}"
        expected = [output_dir / "run_parts" / f"{prefix}.{kind}.csv" for kind in ("candidates", "methods", "global")]
        if all(path.exists() for path in expected):
            continue
        run_one(
            dataset,
            seed,
            mode,
            fraction,
            data_root=Path(args.data_root),
            source=source_dir(dataset, root, overrides),
            output_dir=output_dir,
            device_name=args.device,
            backbones=tuple(args.backbones),
            constructors=tuple(args.constructors),
            ratios=tuple(args.ratios),
            downstream_epochs=args.downstream_epochs,
            auxiliary_epochs=args.auxiliary_epochs,
            vgae_epochs=args.vgae_epochs,
            gate_margin_pp=args.gate_margin_pp,
        )
    if args.finalize:
        aggregate(output_dir)


if __name__ == "__main__":
    main()

