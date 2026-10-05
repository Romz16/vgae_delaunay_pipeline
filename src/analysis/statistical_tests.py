"""Statistical tests for validation-selected rewiring results."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_test_rows(selected_runs: pd.DataFrame) -> pd.DataFrame:
    """Compute paired tests between baseline and selected rewiring per dataset/backbone.

    Expected columns include dataset, backbone, seed, baseline_test_f1 and
    selected_test_f1. Values are assumed to be in [0, 1].
    """
    if selected_runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = ["dataset", "backbone"]
    for (dataset, backbone), group in selected_runs.groupby(group_cols):
        group = group.dropna(subset=["baseline_test_f1", "selected_test_f1"])
        if group.empty:
            continue
        baseline = group["baseline_test_f1"].to_numpy(dtype=float)
        selected = group["selected_test_f1"].to_numpy(dtype=float)
        diff = selected - baseline
        n = int(diff.size)
        mean_diff = float(np.mean(diff)) if n else float("nan")
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        ci95 = float(1.96 * std_diff / np.sqrt(n)) if n > 1 else 0.0
        cohen_dz = float(mean_diff / std_diff) if std_diff > 0 else float("nan")

        try:
            wilcoxon = stats.wilcoxon(selected, baseline, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except Exception:
            wilcoxon_stat = float("nan")
            wilcoxon_p = float("nan")

        try:
            ttest = stats.ttest_rel(selected, baseline, nan_policy="omit")
            paired_t_stat = float(ttest.statistic)
            paired_t_p = float(ttest.pvalue)
        except Exception:
            paired_t_stat = float("nan")
            paired_t_p = float("nan")

        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "n": n,
                "mean_baseline_f1": float(np.mean(baseline)) if n else float("nan"),
                "mean_selected_f1": float(np.mean(selected)) if n else float("nan"),
                "mean_difference": mean_diff,
                "ci95_difference_lower": mean_diff - ci95,
                "ci95_difference_upper": mean_diff + ci95,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "paired_t_statistic": paired_t_stat,
                "paired_t_p": paired_t_p,
                "cohen_dz": cohen_dz,
            }
        )
    return pd.DataFrame(rows)
