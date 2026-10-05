#!/usr/bin/env python
"""CLI for the controlled synthetic rewiring suitability study."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from any working directory without requiring callers
# to manage PYTHONPATH explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_suitability.config import SyntheticExperimentConfig, profile_config


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("init","catalog","run","aggregate","analyze-recovery","fit","validate-real","report","all"))
    parser.add_argument("--profile",choices=("smoke","pilot","full","recovery_smoke","recovery_pilot","recovery_full","blind_wave1","blind_wave2","blind_wave3"),default="pilot")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--max-graphs",type=int)
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--num-shards",type=int,default=1)
    parser.add_argument("--no-finalize",action="store_true")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--real-metrics",type=Path,default=Path("outputs_revision/downloaded_results/global_12datasets_corrected_20260824/original_graph_metrics.csv"))
    parser.add_argument("--real-gains",type=Path,default=Path("outputs_revision/downloaded_results/global_12datasets_corrected_20260824/exploratory_rewiring_recommendations.csv"))
    return parser.parse_args()


def main():
    args=parse_args()
    config=SyntheticExperimentConfig.load(args.config) if args.config else profile_config(args.profile,args.output_dir)
    config.output_dir.mkdir(parents=True,exist_ok=True)
    config.save(config.output_dir/"experiment_config.json")
    from src.synthetic_suitability.experiment import design_summary
    summary=design_summary(config)
    print(summary)
    if args.phase=="init": return
    if args.phase in {"catalog","all"}:
        from src.synthetic_suitability.experiment import graph_catalog
        graph_catalog(config,overwrite=args.overwrite)
        if args.phase=="catalog": return
    if args.phase in {"run","all"}:
        from src.synthetic_suitability.experiment import run_experiments
        run_experiments(
            config,
            resume=not args.overwrite,
            max_graphs=args.max_graphs,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            finalize=not args.no_finalize,
        )
        if args.phase=="run": return
    if args.phase=="aggregate":
        from src.synthetic_suitability.experiment import aggregate_outcomes
        aggregate_outcomes(config); return
    if args.phase=="analyze-recovery":
        from src.synthetic_suitability.recovery_analysis import analyze_recovery
        analyze_recovery(config.output_dir); return
    if args.phase in {"fit","all"}:
        from src.synthetic_suitability.indicator import fit_indicator
        fit_indicator(config)
        if args.phase=="fit": return
    if args.phase in {"validate-real","all"}:
        from src.synthetic_suitability.indicator import validate_on_real_datasets
        validate_on_real_datasets(config,args.real_metrics,args.real_gains)
        if args.phase=="validate-real": return
    if args.phase in {"report","all"}:
        from src.synthetic_suitability.reporting import generate_outputs
        generate_outputs(config)


if __name__=="__main__":
    main()
