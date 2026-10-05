"""Cluster-safe analysis for the latent corruption/recovery experiment."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def analyze_recovery(output_dir: Path, seed: int = 20260917, bootstrap: int = 2000) -> pd.DataFrame:
    executions = pd.read_csv(output_dir / "all_execution_results.csv")
    keys = ["graph_id", "task_id", "pipeline_seed"]
    original = executions[executions.constructor.eq("original")][keys + ["test_f1"]].rename(
        columns={"test_f1": "original_test_f1"}
    )
    compared = executions.merge(original, on=keys, how="left")
    compared["gain_pp"] = 100.0 * (compared.test_f1 - compared.original_test_f1)
    compared = compared[~compared.constructor.eq("original")].copy()

    # The graph instance is the inferential unit. Pipeline and feature seeds
    # are repeated measurements and are averaged before any hypothesis test.
    unit_cols = ["graph_id", "constructor", "ratio"]
    units = compared.groupby(unit_cols, dropna=False, as_index=False).agg(gain_pp=("gain_pp", "mean"))
    rows = []
    rng = np.random.default_rng(seed)
    for (constructor, ratio), part in units.groupby(["constructor", "ratio"], dropna=False):
        values = part.gain_pp.to_numpy(float)
        statistic, pvalue = _wilcoxon(values)
        lo, hi = _bootstrap_mean(values, rng, bootstrap)
        rows.append({
            "constructor": constructor, "ratio": ratio, "n_graphs": len(values),
            "mean_gain_pp": float(np.mean(values)), "median_gain_pp": float(np.median(values)),
            "ci95_low_pp": lo, "ci95_high_pp": hi,
            "positive_graph_fraction": float(np.mean(values > 0)),
            "wilcoxon_statistic": statistic, "wilcoxon_p": pvalue,
        })
    overall = pd.DataFrame(rows)
    overall["wilcoxon_p_holm"] = _holm(overall.wilcoxon_p.to_numpy(float))
    overall.to_csv(output_dir / "recovery_primary_comparisons.csv", index=False)

    strata = []
    dimensions = ["param_corruption_fraction", "param_corruption_mode", "param_latent_homophily",
                  "param_n", "param_target_avg_degree", "feature_regime"]
    for dimension in dimensions:
        for (level, constructor, ratio), part in compared.groupby(
            [dimension, "constructor", "ratio"], dropna=False
        ):
            graph_values = part.groupby("graph_id").gain_pp.mean().to_numpy(float)
            lo, hi = _bootstrap_mean(graph_values, rng, bootstrap)
            _, pvalue = _wilcoxon(graph_values)
            strata.append({
                "dimension": dimension, "level": level, "constructor": constructor, "ratio": ratio,
                "n_graphs": len(graph_values), "mean_gain_pp": float(np.mean(graph_values)),
                "median_gain_pp": float(np.median(graph_values)), "ci95_low_pp": lo,
                "ci95_high_pp": hi, "positive_graph_fraction": float(np.mean(graph_values > 0)),
                "wilcoxon_p": pvalue,
            })
    pd.DataFrame(strata).to_csv(output_dir / "recovery_stratified_comparisons.csv", index=False)
    _write_summary(output_dir, overall)
    return overall


def _wilcoxon(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0):
        return 0.0, 1.0
    result = wilcoxon(values, zero_method="pratt", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def _bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _holm(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues); adjusted = np.empty_like(pvalues)
    running = 0.0; m = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _write_summary(output_dir: Path, overall: pd.DataFrame) -> None:
    ranked = overall.sort_values("mean_gain_pp", ascending=False)
    lines = [
        "# Corruption–recovery: primary statistical analysis", "",
        "The independent graph instance is the inferential unit. Feature/pipeline repetitions are averaged ",
        "before the paired Wilcoxon test; confidence intervals use a graph-level bootstrap. Holm correction ",
        "controls multiplicity across constructors and ratios.", "", "| Constructor | Ratio | Mean gain (pp) | 95% CI | Positive graphs | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranked.itertuples():
        ratio = "—" if pd.isna(row.ratio) else f"{row.ratio:.2f}"
        lines.append(
            f"| {row.constructor} | {ratio} | {row.mean_gain_pp:.3f} | "
            f"[{row.ci95_low_pp:.3f}, {row.ci95_high_pp:.3f}] | "
            f"{100 * row.positive_graph_fraction:.1f}% | {row.wilcoxon_p_holm:.4g} |"
        )
    (output_dir / "RECOVERY_STATISTICAL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
