#!/usr/bin/env python3
"""Backfill the Raw features -> UMAP -> Delaunay ablation for a completed run."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DelaunayConfig, OptunaConfig, OutputConfig, PipelineConfig, RewiringConfig, SplitConfig, TrainingConfig
from src.analysis.structural_metrics import StructuralMetricConfig, compute_structural_metrics
from src.data.loaders import GraphLoader
from src.data.splits import apply_node_split
from src.models.gnn import GNNParams
from src.pipeline.orchestrator import VGAEDelaunayPipeline
from src.rewiring.delaunay_builder import DelaunayGraphBuilder
from src.utils.device import resolve_device
from src.utils.logging_utils import configure_logging


def load_pipeline_config(output_dir: Path, device_name: str) -> PipelineConfig:
    payload = json.loads((output_dir / "pipeline_config.json").read_text(encoding="utf-8"))
    output_payload = dict(payload["output"])
    output_payload["root_dir"] = output_dir
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
        delaunay=DelaunayConfig(**payload["delaunay"]),
        output=OutputConfig(**output_payload),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root", default="./data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    output_dir = Path(args.output_dir)
    config = load_pipeline_config(output_dir, args.device)
    if config.dataset_name.lower() != args.dataset.lower():
        raise ValueError(f"Dataset mismatch: config={config.dataset_name}, requested={args.dataset}")
    device = resolve_device(args.device)
    data = GraphLoader(root=args.root, device=device).load_from_name(args.dataset)
    data = apply_node_split(data, seed=config.base_seed, config=config.split, device=device)

    builder = DelaunayGraphBuilder(config, device)
    raw_edge_index, raw_features, raw_points = builder.build_from_raw_features_with_projection(data)
    pipeline = VGAEDelaunayPipeline(config, device)
    pipeline._write_umap_trustworthiness(builder, [("raw_features", raw_features, raw_points)], append=True)

    metrics_path = config.output.revision_dir() / "selected_rewired_graph_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[~metrics["graph_name"].eq("Raw features -> UMAP -> Delaunay")]
    raw_metric = compute_structural_metrics(
        data=data,
        dataset=config.dataset_name,
        graph_name="Raw features -> UMAP -> Delaunay",
        config=StructuralMetricConfig(seed=config.base_seed),
        edge_index=raw_edge_index,
    )
    pd.concat([metrics, pd.DataFrame([raw_metric])], ignore_index=True).to_csv(metrics_path, index=False)

    params_payload = json.loads((output_dir / "best_gnn_params.json").read_text(encoding="utf-8"))
    params = {name: GNNParams(**values) for name, values in params_payload.items()}
    pipeline._evaluate_raw_feature_baseline(data, params, raw_edge_index)
    print(f"Raw-feature ablation saved to {config.output.revision_dir()}")


if __name__ == "__main__":
    main()
