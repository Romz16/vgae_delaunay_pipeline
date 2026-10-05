#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.statistical_tests import paired_test_rows
from src.analysis.structural_correlations import (
    baseline_saturation_analysis,
    dataset_level_gain,
    density_confounder_analysis,
    exploratory_recommendations,
    leave_one_out_sensitivity,
    old_vs_corrected,
    PRIMARY_METRICS,
    safe_correlation,
    structural_metric_correlations,
)


def collect_csv(
    roots: list[Path],
    filename: str,
    *,
    recover_dataset: bool = False,
) -> pd.DataFrame:
    frames = []
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob(filename):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                frame = pd.read_csv(path)
                if recover_dataset and "dataset" not in frame.columns:
                    companion = path.with_name("selected_rates.csv")
                    if companion.exists():
                        datasets = pd.read_csv(companion, usecols=["dataset"])["dataset"].dropna().unique()
                        if len(datasets) == 1:
                            frame.insert(0, "dataset", datasets[0])
                frames.append(frame)
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def selected_before_after(metrics: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or selected.empty or "selected_condition" not in selected.columns:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in dict.fromkeys(
            [
                "num_nodes",
                "num_edges",
                "density",
                "avg_degree",
                "degree_cv",
                "edge_homophily",
                "lambda2_norm_laplacian",
                "cheeger_lower_bound",
                "cheeger_upper_bound",
                "approx_diameter",
                "approx_avg_shortest_path",
                "effective_resistance_sample",
                "num_components",
                "largest_cc_ratio",
                "num_isolates",
            ]
            + PRIMARY_METRICS
        )
        if column in metrics.columns
    ]
    original = metrics[metrics["graph_name"].eq("Original")][["dataset", *metric_columns]].drop_duplicates("dataset")
    original = original.rename(columns={column: f"original_{column}" for column in metric_columns})
    rewired = metrics[~metrics["graph_name"].eq("Original")][["dataset", "graph_name", *metric_columns]]
    rewired = rewired.drop_duplicates(["dataset", "graph_name"])
    rewired = rewired.rename(columns={column: f"selected_{column}" for column in metric_columns})
    merged = selected.merge(original, on="dataset", how="left")
    merged = merged.merge(
        rewired,
        left_on=["dataset", "selected_condition"],
        right_on=["dataset", "graph_name"],
        how="left",
    )
    for column in metric_columns:
        merged[f"delta_{column}"] = merged[f"selected_{column}"] - merged[f"original_{column}"]
    return merged


def delta_correlations(before_after: pd.DataFrame) -> pd.DataFrame:
    if before_after.empty:
        return pd.DataFrame()
    delta_columns = [column for column in before_after.columns if column.startswith("delta_")]
    aggregations = {column: "mean" for column in delta_columns}
    aggregations["gain_f1"] = "mean"
    dataset_level = before_after.groupby("dataset", as_index=False).agg(aggregations)
    rows = []
    for column in delta_columns:
        corr = safe_correlation(dataset_level[column], dataset_level["gain_f1"])
        if corr is not None:
            rows.append({"metric": column, "target": "corrected_mean_gain", **corr})
    return pd.DataFrame(rows)


def raw_feature_comparison(raw_runs: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if raw_runs.empty or selected.empty:
        return pd.DataFrame()
    baseline_columns = [
        "dataset",
        "backbone",
        "seed",
        "baseline_test_acc",
        "baseline_test_f1",
    ]
    baseline = selected[baseline_columns].drop_duplicates(["dataset", "backbone", "seed"])
    merged = raw_runs.merge(baseline, on=["dataset", "backbone", "seed"], how="left")
    merged["raw_gain_acc_vs_original"] = merged["test_acc"] - merged["baseline_test_acc"]
    merged["raw_gain_f1_vs_original"] = merged["test_f1"] - merged["baseline_test_f1"]
    return merged


def write_global_figures(metrics: pd.DataFrame, selected: pd.DataFrame, before_after: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    gain = dataset_level_gain(selected)
    original = metrics[metrics["graph_name"].eq("Original")].drop_duplicates("dataset")
    merged = gain.merge(original, on="dataset", how="left")
    plots = [
        ("lambda2_norm_laplacian", "corrected_mean_gain", "lambda2_vs_corrected_gain.png"),
        ("cheeger_upper_bound", "corrected_mean_gain", "cheeger_upper_vs_corrected_gain.png"),
        ("density", "corrected_mean_gain", "density_vs_corrected_gain.png"),
        ("edge_homophily", "corrected_mean_gain", "homophily_vs_corrected_gain.png"),
        ("baseline_f1_mean", "corrected_mean_gain", "baseline_vs_corrected_gain.png"),
    ]
    for x, y, filename in plots:
        if x not in merged.columns or y not in merged.columns:
            continue
        frame = merged[["dataset", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            continue
        plt.figure(figsize=(7, 5))
        plt.scatter(frame[x], frame[y])
        for _, row in frame.iterrows():
            plt.annotate(str(row["dataset"]), (row[x], row[y]), fontsize=7)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=200)
        plt.close()

    if before_after.empty:
        return
    delta_columns = [
        column
        for column in ["delta_lambda2_norm_laplacian", "delta_cheeger_upper_bound", "delta_density", "delta_edge_homophily"]
        if column in before_after.columns
    ]
    aggregations = {column: "mean" for column in delta_columns}
    aggregations["gain_f1"] = "mean"
    delta_level = before_after.groupby("dataset", as_index=False).agg(aggregations)
    for column in delta_columns:
        frame = delta_level[["dataset", column, "gain_f1"]].replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            continue
        plt.figure(figsize=(7, 5))
        plt.scatter(frame[column], frame["gain_f1"])
        for _, row in frame.iterrows():
            plt.annotate(str(row["dataset"]), (row[column], row["gain_f1"]), fontsize=7)
        plt.xlabel(column)
        plt.ylabel("corrected_mean_gain")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{column}_vs_corrected_gain.png", dpi=200)
        plt.close()


def preliminary_correlations() -> pd.DataFrame:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        nargs="+",
        default=["outputs"],
        help="One or more roots containing per-dataset outputs_revision folders",
    )
    parser.add_argument("--out", default="outputs_revision_global", help="Output folder for global revision analysis")
    args = parser.parse_args()

    roots = [Path(root) for root in args.root]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    metrics = collect_csv(roots, "selected_rewired_graph_metrics.csv")
    selected = collect_csv(roots, "selected_rates.csv")
    all_runs = collect_csv(roots, "all_runs_long.csv")
    final_results = collect_csv(roots, "final_selected_test_results.csv", recover_dataset=True)
    raw_results = collect_csv(roots, "raw_feature_delaunay_results.csv")
    raw_runs = collect_csv(roots, "raw_feature_delaunay_all_runs.csv")
    trustworthiness = collect_csv(roots, "umap_trustworthiness.csv")

    metrics.to_csv(out / "selected_rewired_graph_metrics.csv", index=False)
    selected.to_csv(out / "selected_rates.csv", index=False)
    all_runs.to_csv(out / "all_runs_long.csv", index=False)
    final_results.to_csv(out / "final_selected_test_results.csv", index=False)
    raw_results.to_csv(out / "raw_feature_delaunay_results.csv", index=False)
    raw_runs.to_csv(out / "raw_feature_delaunay_all_runs.csv", index=False)
    trustworthiness.to_csv(out / "umap_trustworthiness.csv", index=False)

    original_metrics = metrics[metrics["graph_name"].eq("Original")] if (not metrics.empty and "graph_name" in metrics.columns) else pd.DataFrame()
    original_metrics.to_csv(out / "original_graph_metrics.csv", index=False)

    paired_test_rows(selected).to_csv(out / "paired_statistical_tests.csv", index=False)
    correlations = structural_metric_correlations(metrics, selected)
    correlations.to_csv(out / "structural_metric_correlations.csv", index=False)
    leave_one_out_sensitivity(metrics, selected).to_csv(out / "structural_correlation_sensitivity.csv", index=False)
    density_confounder_analysis(metrics, selected).to_csv(out / "density_confounder_analysis.csv", index=False)
    baseline_saturation_analysis(metrics, selected).to_csv(out / "baseline_saturation_analysis.csv", index=False)
    old_vs_corrected(preliminary_correlations(), correlations).to_csv(
        out / "structural_correlations_old_vs_corrected.csv", index=False
    )
    exploratory_recommendations(metrics, dataset_level_gain(selected)).to_csv(
        out / "exploratory_rewiring_recommendations.csv", index=False
    )
    before_after = selected_before_after(metrics, selected)
    before_after.to_csv(out / "selected_before_after_structural_metrics.csv", index=False)
    delta_correlations(before_after).to_csv(out / "structural_delta_correlations.csv", index=False)
    raw_feature_comparison(raw_runs, selected).to_csv(out / "raw_feature_delaunay_comparison.csv", index=False)
    write_global_figures(metrics, selected, before_after, out / "figures")

    dataset_names = sorted(selected["dataset"].dropna().astype(str).unique()) if "dataset" in selected else []
    effective_resistance_missing = (
        int(metrics["effective_resistance_sample"].isna().sum())
        if "effective_resistance_sample" in metrics
        else 0
    )
    summary = [
        "# Global Revision Aggregation",
        "",
        f"Datasets found in selected_runs: {len(dataset_names)}",
        f"Datasets: {', '.join(dataset_names)}",
        f"Rows in all_runs_long: {len(all_runs)}",
        f"Rows in selected_rates: {len(selected)}",
        f"Rows in final_selected_test_results: {len(final_results)}",
        f"Datasets in final_selected_test_results: {final_results['dataset'].nunique() if 'dataset' in final_results else 0}",
        f"Rows in raw_feature_delaunay_all_runs: {len(raw_runs)}",
        f"Rows in selected_before_after_structural_metrics: {len(before_after)}",
        f"UMAP trustworthiness rows: {len(trustworthiness)}",
        f"Missing effective_resistance_sample values in structural metrics: {effective_resistance_missing}",
        "",
        "The correlation files are exploratory. Effective-resistance analyses may use fewer datasets when the numerical solver did not converge; the primary lambda2, Cheeger, density and homophily analyses are unaffected.",
    ]
    (out / "REVISION_EXPERIMENT_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"Global revision outputs saved to {out}")


if __name__ == "__main__":
    main()
