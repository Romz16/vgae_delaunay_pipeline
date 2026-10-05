"""Correlation and exploratory diagnostics for structural rewiring analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PRIMARY_METRICS = [
    "lambda2_norm_laplacian",
    "cheeger_upper_bound",
    "density",
    "edge_homophily",
    "approx_diameter",
    "approx_avg_shortest_path",
    "effective_resistance_sample",
    "degree_cv",
]


def safe_correlation(x, y):
    """Return Pearson/Spearman correlations or None if unavailable."""
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    pr, pp = pearsonr(frame["x"], frame["y"])
    sr, sp = spearmanr(frame["x"], frame["y"])
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "n": int(len(frame)),
    }


def dataset_level_gain(selected_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation-selected gains to dataset level.

    The primary aggregation is the predeclared mean across backbones.
    """
    if selected_runs.empty:
        return pd.DataFrame()
    per_backbone = (
        selected_runs.groupby(["dataset", "backbone"], as_index=False)
        .agg(
            baseline_f1_mean=("baseline_test_f1", "mean"),
            selected_f1_mean=("selected_test_f1", "mean"),
            corrected_gain_mean=("gain_f1", "mean"),
            selected_ratio_mode=("selected_ratio", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        )
    )
    dataset_level = (
        per_backbone.groupby("dataset", as_index=False)
        .agg(
            baseline_f1_mean=("baseline_f1_mean", "mean"),
            selected_f1_mean=("selected_f1_mean", "mean"),
            corrected_mean_gain=("corrected_gain_mean", "mean"),
            corrected_best_backbone_gain=("corrected_gain_mean", "max"),
            corrected_worst_backbone_gain=("corrected_gain_mean", "min"),
        )
    )
    return dataset_level


def structural_metric_correlations(metrics_df: pd.DataFrame, selected_runs: pd.DataFrame) -> pd.DataFrame:
    """Compute main dataset-level structural correlations."""
    dataset_gain = dataset_level_gain(selected_runs)
    if dataset_gain.empty or metrics_df.empty:
        return pd.DataFrame()
    original_metrics = metrics_df[metrics_df["graph_name"].eq("Original")].copy()
    merged = dataset_gain.merge(original_metrics, on="dataset", how="left")

    rows: list[dict[str, object]] = []
    targets = ["corrected_mean_gain", "corrected_best_backbone_gain", "baseline_f1_mean"]
    for metric, target in itertools.product(PRIMARY_METRICS, targets):
        if metric not in merged.columns or target not in merged.columns:
            continue
        corr = safe_correlation(merged[metric], merged[target])
        if corr is None:
            continue
        rows.append({"metric": metric, "target": target, **corr})
    return pd.DataFrame(rows)


def leave_one_out_sensitivity(metrics_df: pd.DataFrame, selected_runs: pd.DataFrame) -> pd.DataFrame:
    """Run leave-one-out sensitivity for primary structural correlations."""
    dataset_gain = dataset_level_gain(selected_runs)
    if dataset_gain.empty or metrics_df.empty:
        return pd.DataFrame()
    original_metrics = metrics_df[metrics_df["graph_name"].eq("Original")].copy()
    merged = dataset_gain.merge(original_metrics, on="dataset", how="left")

    rows: list[dict[str, object]] = []
    exclusions = [(None, "all_datasets")]
    exclusions.extend((dataset, f"without_{dataset}") for dataset in merged["dataset"].dropna().unique())
    exclusions.extend(("Roman-Empire", "without_Roman-Empire") for _ in [])

    for excluded_dataset, label in exclusions:
        sub = merged if excluded_dataset is None else merged[merged["dataset"] != excluded_dataset]
        for metric in PRIMARY_METRICS:
            if metric not in sub.columns:
                continue
            corr = safe_correlation(sub[metric], sub["corrected_mean_gain"])
            if corr is None:
                continue
            rows.append({"sensitivity_scope": label, "excluded_dataset": excluded_dataset or "", "metric": metric, **corr})

    # Explicit combined sensitivity for the two known outliers.
    no_extreme = merged[~merged["dataset"].isin(["Roman-Empire", "Minesweeper"])]
    for metric in PRIMARY_METRICS:
        if metric not in no_extreme.columns:
            continue
        corr = safe_correlation(no_extreme[metric], no_extreme["corrected_mean_gain"])
        if corr is not None:
            rows.append(
                {
                    "sensitivity_scope": "without_Roman-Empire_and_Minesweeper",
                    "excluded_dataset": "Roman-Empire;Minesweeper",
                    "metric": metric,
                    **corr,
                }
            )
    return pd.DataFrame(rows)


def old_vs_corrected(old_correlations: pd.DataFrame | None, corrected: pd.DataFrame) -> pd.DataFrame:
    """Compare preliminary and corrected structural correlations when old values exist."""
    if old_correlations is None or old_correlations.empty or corrected.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "old_pearson_r",
                "old_pearson_p",
                "old_spearman_rho",
                "old_spearman_p",
                "corrected_pearson_r",
                "corrected_pearson_p",
                "corrected_spearman_rho",
                "corrected_spearman_p",
                "direction_preserved",
                "significance_preserved",
            ]
        )
    old = old_correlations.rename(
        columns={
            "pearson_r": "old_pearson_r",
            "pearson_p": "old_pearson_p",
            "spearman_rho": "old_spearman_rho",
            "spearman_p": "old_spearman_p",
        }
    )
    new = corrected[corrected["target"].eq("corrected_mean_gain")].rename(
        columns={
            "pearson_r": "corrected_pearson_r",
            "pearson_p": "corrected_pearson_p",
            "spearman_rho": "corrected_spearman_rho",
            "spearman_p": "corrected_spearman_p",
        }
    )
    merged = old.merge(new, on="metric", how="outer")
    merged["direction_preserved"] = np.sign(merged["old_pearson_r"]) == np.sign(merged["corrected_pearson_r"])
    merged["significance_preserved"] = (merged["old_pearson_p"] < 0.05) & (merged["corrected_pearson_p"] < 0.05)
    cols = [
        "metric",
        "old_pearson_r",
        "old_pearson_p",
        "old_spearman_rho",
        "old_spearman_p",
        "corrected_pearson_r",
        "corrected_pearson_p",
        "corrected_spearman_rho",
        "corrected_spearman_p",
        "direction_preserved",
        "significance_preserved",
    ]
    return merged[[c for c in cols if c in merged.columns]]


def exploratory_recommendations(metrics_df: pd.DataFrame, dataset_gain: pd.DataFrame | None = None) -> pd.DataFrame:
    """Produce descriptive, non-final rewiring recommendations."""
    original = metrics_df[metrics_df["graph_name"].eq("Original")].copy()
    if dataset_gain is not None and not dataset_gain.empty:
        original = original.merge(dataset_gain, on="dataset", how="left")

    def recommend(row):
        baseline = row.get("baseline_f1_mean", np.nan)
        lambda2 = row.get("lambda2_norm_laplacian", np.nan)
        diameter = row.get("approx_diameter", np.nan)
        avg_path = row.get("approx_avg_shortest_path", np.nan)
        if np.isfinite(baseline) and baseline >= 0.88:
            return "not_recommended_baseline_saturated"
        if np.isfinite(lambda2) and lambda2 < 0.001:
            return "high_risk_very_low_spectral_gap"
        if (np.isfinite(diameter) and diameter > 50) or (np.isfinite(avg_path) and avg_path > 20):
            return "high_risk_long_range_bottleneck"
        if np.isfinite(lambda2) and np.isfinite(avg_path) and lambda2 >= 0.03 and avg_path <= 4:
            return "recommended"
        if np.isfinite(lambda2) and lambda2 >= 0.01:
            return "test_with_validation"
        return "uncertain"

    original["exploratory_recommendation"] = original.apply(recommend, axis=1)
    original["recommendation_scope"] = "descriptive_only_not_a_validated_rule"
    return original


def density_confounder_analysis(metrics_df: pd.DataFrame, selected_runs: pd.DataFrame) -> pd.DataFrame:
    """Small exploratory analysis of density as a confounder.

    Uses simple correlations only, because N=12 is too small for a rich model.
    """
    dataset_gain = dataset_level_gain(selected_runs)
    original = metrics_df[metrics_df["graph_name"].eq("Original")].copy()
    merged = dataset_gain.merge(original, on="dataset", how="left")
    pairs = [
        ("density", "corrected_mean_gain"),
        ("density", "lambda2_norm_laplacian"),
        ("density", "cheeger_upper_bound"),
        ("lambda2_norm_laplacian", "corrected_mean_gain"),
        ("cheeger_upper_bound", "corrected_mean_gain"),
    ]
    rows: list[dict[str, object]] = []
    for x, y in pairs:
        if x not in merged.columns or y not in merged.columns:
            continue
        corr = safe_correlation(merged[x], merged[y])
        if corr is not None:
            rows.append({"analysis": "bivariate", "x": x, "y": y, "control": "", **corr})

    for x in ["lambda2_norm_laplacian", "cheeger_upper_bound"]:
        needed = [x, "corrected_mean_gain", "density"]
        if not all(column in merged.columns for column in needed):
            continue
        frame = merged[needed].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 4 or frame["density"].nunique() < 2:
            continue
        design = np.column_stack([np.ones(len(frame)), frame["density"].to_numpy(dtype=float)])
        x_values = frame[x].to_numpy(dtype=float)
        y_values = frame["corrected_mean_gain"].to_numpy(dtype=float)
        x_residual = x_values - design @ np.linalg.lstsq(design, x_values, rcond=None)[0]
        y_residual = y_values - design @ np.linalg.lstsq(design, y_values, rcond=None)[0]
        corr = safe_correlation(x_residual, y_residual)
        if corr is not None:
            rows.append(
                {
                    "analysis": "partial_controlling_density",
                    "x": x,
                    "y": "corrected_mean_gain",
                    "control": "density",
                    **corr,
                }
            )
    return pd.DataFrame(rows)


def baseline_saturation_analysis(metrics_df: pd.DataFrame, selected_runs: pd.DataFrame) -> pd.DataFrame:
    """Correlate baseline performance with corrected rewiring gain."""
    dataset_gain = dataset_level_gain(selected_runs)
    if dataset_gain.empty:
        return pd.DataFrame()
    corr = safe_correlation(dataset_gain["baseline_f1_mean"], dataset_gain["corrected_mean_gain"])
    if corr is None:
        return pd.DataFrame()
    return pd.DataFrame([{"x": "baseline_f1_mean", "y": "corrected_mean_gain", **corr}])
