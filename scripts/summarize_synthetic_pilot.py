#!/usr/bin/env python
"""Independent quantitative audit of the synthetic suitability pilot outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


STRUCTURAL_METRICS = [
    "lambda2_norm_laplacian", "density", "avg_degree", "degree_cv",
    "avg_clustering", "degree_assortativity", "approx_avg_shortest_path",
    "approx_diameter", "effective_resistance_sample", "edge_connectivity_ratio",
    "spectral_sweep_conductance", "modularity",
]


def grouped_bootstrap_ci(frame: pd.DataFrame, value: str, repetitions: int = 20_000) -> list[float]:
    rng = np.random.default_rng(20260917)
    groups = {
        key: part[value].to_numpy(dtype=float)
        for key, part in frame.groupby("configuration_id", sort=False)
    }
    keys = np.asarray(list(groups), dtype=object)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        estimates[index] = np.concatenate([groups[key] for key in sampled]).mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def effect_summary(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().to_numpy(dtype=float)
    test = wilcoxon(clean, alternative="two-sided", zero_method="wilcox")
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "q025": float(np.quantile(clean, 0.025)),
        "q975": float(np.quantile(clean, 0.975)),
        "positive_fraction": float(np.mean(clean > 0)),
        "success_0_5pp_fraction": float(np.mean(clean > 0.005)),
        "success_1_0pp_fraction": float(np.mean(clean > 0.01)),
        "wilcoxon_statistic": float(test.statistic),
        "wilcoxon_p_two_sided": float(test.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results

    graph = pd.read_csv(out / "graph_level_outcomes.csv")
    selected = pd.read_csv(out / "selected_execution_results.csv")
    all_runs = pd.read_csv(out / "all_execution_results.csv")
    external = pd.read_csv(out / "external_real_dataset_validation.csv")
    regression = pd.read_csv(out / "gain_regression_metrics.csv")

    summary: dict[str, object] = {
        "independent_graphs": int(len(graph)),
        "configurations": int(graph["configuration_id"].nunique()),
        "families": int(graph["family"].nunique()),
        "nested_runs": int(len(selected)),
        "conditions": int(len(all_runs)),
    }
    for name, column in {
        "selected_rewiring": "gain_selected_rewiring",
        "geometry_only": "gain_geometry",
        "vgae_increment": "gain_vgae",
    }.items():
        block = effect_summary(graph[column])
        block["configuration_bootstrap_mean_ci95"] = grouped_bootstrap_ci(graph, column)
        summary[name] = block

    family = graph.groupby("family", as_index=False).agg(
        graphs=("graph_id", "size"),
        configurations=("configuration_id", "nunique"),
        mean_selected_gain=("gain_selected_rewiring", "mean"),
        median_selected_gain=("gain_selected_rewiring", "median"),
        mean_geometry_gain=("gain_geometry", "mean"),
        mean_vgae_increment=("gain_vgae", "mean"),
        positive_fraction=("gain_selected_rewiring", lambda x: float(np.mean(x > 0))),
        success_0_5pp_fraction=("gain_selected_rewiring", lambda x: float(np.mean(x > 0.005))),
    ).sort_values("mean_selected_gain", ascending=False)
    family.to_csv(out / "analysis_family_summary.csv", index=False)

    graph_regime = selected.groupby(
        ["graph_id", "family", "configuration_id", "feature_regime"], as_index=False
    ).agg(
        selected_gain=("gain_selected_rewiring", "mean"),
        geometry_gain=("gain_geometry", "mean"),
        vgae_increment=("gain_vgae", "mean"),
        baseline_f1=("baseline_test_f1", "mean"),
    )
    regime = graph_regime.groupby("feature_regime", as_index=False).agg(
        graph_regimes=("graph_id", "size"),
        mean_baseline_f1=("baseline_f1", "mean"),
        mean_selected_gain=("selected_gain", "mean"),
        median_selected_gain=("selected_gain", "median"),
        mean_geometry_gain=("geometry_gain", "mean"),
        mean_vgae_increment=("vgae_increment", "mean"),
        positive_fraction=("selected_gain", lambda x: float(np.mean(x > 0))),
        success_0_5pp_fraction=("selected_gain", lambda x: float(np.mean(x > 0.005))),
    ).sort_values("mean_selected_gain", ascending=False)
    regime.to_csv(out / "analysis_feature_regime_summary.csv", index=False)

    keys = ["graph_id", "task_id", "pipeline_seed"]
    baseline = all_runs.loc[all_runs["constructor"].eq("original"), keys + ["test_f1"]].rename(
        columns={"test_f1": "original_test_f1"}
    )
    conditions = all_runs.merge(baseline, on=keys, how="left", validate="many_to_one")
    conditions["gain_vs_original"] = conditions["test_f1"] - conditions["original_test_f1"]
    constructor = conditions.groupby(["stage", "constructor", "ratio"], dropna=False, as_index=False).agg(
        runs=("test_f1", "size"),
        mean_test_f1=("test_f1", "mean"),
        mean_gain=("gain_vs_original", "mean"),
        median_gain=("gain_vs_original", "median"),
        positive_fraction=("gain_vs_original", lambda x: float(np.mean(x > 0))),
        mean_training_seconds=("training_seconds", "mean"),
        mean_rewiring_seconds=("rewiring_seconds", "mean"),
    ).sort_values(["stage", "constructor", "ratio"], na_position="first")
    constructor.to_csv(out / "analysis_constructor_summary.csv", index=False)

    ratios = selected.groupby("selected_ratio", as_index=False).agg(
        runs=("run_id", "size"),
        fraction=("run_id", lambda x: len(x) / len(selected)),
        mean_selected_gain=("gain_selected_rewiring", "mean"),
    ).sort_values("selected_ratio")
    ratios.to_csv(out / "analysis_selected_ratio_summary.csv", index=False)

    correlation_rows = []
    for metric in STRUCTURAL_METRICS:
        clean = graph[[metric, "gain_selected_rewiring"]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, pvalue = spearmanr(clean[metric], clean["gain_selected_rewiring"])
        correlation_rows.append({"metric": metric, "n": len(clean), "spearman_rho": rho, "pvalue": pvalue})
    correlations = pd.DataFrame(correlation_rows).sort_values("spearman_rho", key=lambda x: x.abs(), ascending=False)
    correlations.to_csv(out / "analysis_structural_correlations.csv", index=False)

    synthetic_ranges = {
        feature: [float(graph[feature].min()), float(graph[feature].max())]
        for feature in ("lambda2_norm_laplacian", "density")
    }
    real_ranges = {
        feature: [float(external[feature].min()), float(external[feature].max())]
        for feature in ("lambda2_norm_laplacian", "density")
    }
    in_range = np.ones(len(external), dtype=bool)
    for feature, (low, high) in synthetic_ranges.items():
        in_range &= external[feature].between(low, high).to_numpy()

    summary["selected_ratio_counts"] = {
        str(row.selected_ratio): int(row.runs) for row in ratios.itertuples()
    }
    summary["baseline_gain_spearman"] = float(
        selected[["baseline_test_f1", "gain_selected_rewiring"]].corr(method="spearman").iloc[0, 1]
    )
    summary["gain_regression"] = regression.to_dict(orient="records")
    summary["external_validation"] = {
        "datasets": int(len(external)),
        "observed_successes_0_5pp": int(external["observed_success"].sum()),
        "recommended": int((external["frozen_decision"] == "Recommend").sum()),
        "uncertain": int((external["frozen_decision"] == "Uncertain").sum()),
        "avoided": int((external["frozen_decision"] == "Avoid").sum()),
        "synthetic_feature_ranges": synthetic_ranges,
        "real_feature_ranges": real_ranges,
        "real_datasets_inside_both_synthetic_ranges": int(in_range.sum()),
    }
    summary["integrity"] = {
        "selected_unique_run_ids": int(selected["run_id"].nunique()),
        "expected_conditions_per_run": int(len(all_runs) // len(selected)),
        "missing_test_f1": int(all_runs["test_f1"].isna().sum()),
        "duplicate_selected_run_ids": int(selected["run_id"].duplicated().sum()),
    }

    (out / "analysis_quantitative_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFAMILY\n", family.to_string(index=False))
    print("\nFEATURE REGIME\n", regime.to_string(index=False))
    print("\nCONSTRUCTORS\n", constructor.to_string(index=False))
    print("\nRATIOS\n", ratios.to_string(index=False))
    print("\nCORRELATIONS\n", correlations.to_string(index=False))
    print("\nEXTERNAL\n", external[["dataset", "corrected_mean_gain", "suitability_probability", "frozen_decision", "observed_success"]].to_string(index=False))


if __name__ == "__main__":
    main()
