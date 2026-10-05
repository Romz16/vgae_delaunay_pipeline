#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_protocol_matched_sota_controls import DATASETS, REPRESENTATIVE_DATASETS, constructors_for_dataset


def collect(root: Path, relative: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(root.glob(f"*/{relative}"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby(["dataset", "method", "backbone"]):
        for metric in ("gain_f1", "gain_acc"):
            values = group[metric].dropna().to_numpy()
            try:
                statistic, pvalue = wilcoxon(values) if len(values) else (float("nan"), float("nan"))
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            rows.append({"dataset": keys[0], "method": keys[1], "backbone": keys[2], "metric": metric, "n": len(values), "mean_gain": values.mean() if len(values) else float("nan"), "wilcoxon_statistic": statistic, "wilcoxon_p": pvalue})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--final-runs", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sota_all = collect(args.root, "sota/sota_all_runs.csv")
    sota_selected = collect(args.root, "sota/sota_selected_runs.csv")
    constructors_all = collect(args.root, "constructors/constructor_all_runs.csv")
    constructors_selected = collect(args.root, "constructors/constructor_selected_runs.csv")
    files = {
        "sota_all_runs.csv": sota_all,
        "sota_selected_runs.csv": sota_selected,
        "constructor_all_runs.csv": constructors_all,
        "constructor_selected_runs.csv": constructors_selected,
    }
    for filename, frame in files.items():
        frame.to_csv(args.out / filename, index=False)
    sota_summary = sota_selected.groupby(["dataset", "method", "backbone"], as_index=False).agg(n=("seed", "count"), test_f1_mean=("selected_test_f1", "mean"), test_f1_std=("selected_test_f1", "std"), test_acc_mean=("selected_test_acc", "mean"), test_acc_std=("selected_test_acc", "std"), gain_f1_mean=("gain_f1", "mean"), gain_f1_std=("gain_f1", "std"), gain_acc_mean=("gain_acc", "mean"), gain_acc_std=("gain_acc", "std"))
    constructor_summary = constructors_selected.groupby(["dataset", "method", "backbone"], as_index=False).agg(n=("seed", "count"), test_f1_mean=("selected_test_f1", "mean"), test_f1_std=("selected_test_f1", "std"), test_acc_mean=("selected_test_acc", "mean"), test_acc_std=("selected_test_acc", "std"), gain_f1_mean=("gain_f1", "mean"), gain_f1_std=("gain_f1", "std"), gain_acc_mean=("gain_acc", "mean"), gain_acc_std=("gain_acc", "std"))
    sota_summary.to_csv(args.out / "sota_summary.csv", index=False)
    constructor_summary.to_csv(args.out / "constructor_summary.csv", index=False)
    paired_tests(sota_selected).to_csv(args.out / "sota_paired_tests.csv", index=False)
    paired_tests(constructors_selected).to_csv(args.out / "constructor_paired_tests.csv", index=False)
    constructor_groups = sum(len(constructors_for_dataset(dataset)) * 3 for dataset in DATASETS)
    expected = {
        "sota_all_runs": 12 * 16 * args.final_runs,
        "sota_selected_runs": 12 * 4 * args.final_runs,
        "constructor_all_runs": constructor_groups * 5 * args.final_runs,
        "constructor_selected_runs": constructor_groups * args.final_runs,
    }
    actual = {"sota_all_runs": len(sota_all), "sota_selected_runs": len(sota_selected), "constructor_all_runs": len(constructors_all), "constructor_selected_runs": len(constructors_selected)}
    checks = {name: actual[name] == value for name, value in expected.items()}
    datasets = sorted(set(sota_selected.get("dataset", pd.Series(dtype=str))).union(set(constructors_selected.get("dataset", pd.Series(dtype=str)))))
    payload = {"status": "PASS" if all(checks.values()) and len(datasets) == 12 else "FAIL", "expected_rows": expected, "actual_rows": actual, "checks": checks, "datasets": datasets, "final_runs": args.final_runs}
    (args.out / "VALIDATION_RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / "PROTOCOL_MATCHED_REPORT.md").write_text(
        "# Protocol-matched SOTA and constructor controls\n\n"
        f"Status: **{payload['status']}**\n\n"
        "- Same 12 dataset loaders/versions and stratified 60/20/20 splits.\n"
        f"- {args.final_runs} common seeds starting at 12345; checkpoint and candidate selection by validation macro-F1.\n"
        "- Same saved 30-trial GNN tuning results and 400 final epochs as the proposed method.\n"
        "- SDRF official sparse Balanced-Forman algorithm; candidate step budgets selected on validation.\n"
        "- DiffWire CT evaluated on the official GCN comparison backbone using scalable spectral CTE edge relevance.\n"
        "- kNN, mutual-kNN, random edge-budget, and random degree-matched run on all 12 datasets using identical auxiliary-GCN UMAP coordinates and the exact Delaunay edge budget.\n"
        f"- Radius and MST+kNN are supplementary controls on representative datasets: {', '.join(sorted(REPRESENTATIVE_DATASETS))}.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
