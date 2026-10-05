
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path("outputs_revision/downloaded_results/corruption_recovery_pilot_20260918")
SOURCE = ROOT / "all_execution_results.csv"
SELECTED_SOURCE = ROOT / "selected_execution_results.csv"
OUT_WIDE = ROOT / "validation_gate_1pp_per_seed_wide.csv"
OUT_LONG = ROOT / "validation_gate_1pp_candidates_long.csv"
OUT_SUMMARY = ROOT / "validation_gate_1pp_reproduction_summary.csv"

KEYS = ["graph_id", "task_id", "pipeline_seed"]
MARGIN_PP = 1.0
ELIGIBLE_CONSTRUCTORS = {
    "original",
    "raw_hd_knn",
    "raw_umap_delaunay",
    "learned_hd_knn",
    "delaunay",
    "delaunay_union_original",
}
PREFERENCE = {
    "original": 0,
    "raw_hd_knn": 1,
    "raw_umap_delaunay": 2,
    "learned_hd_knn": 3,
    "delaunay_union_original": 4,
    "delaunay": 5,
}


def candidate_label(constructor: str, ratio: float | None) -> str:
    if constructor == "delaunay" and pd.notna(ratio) and float(ratio) > 0:
        return "vgae"
    return constructor


def best_candidate_table(long_df: pd.DataFrame, candidate: str, prefix: str) -> pd.DataFrame:
    part = long_df[long_df["candidate"].eq(candidate)].copy()
    if part.empty:
        return pd.DataFrame(columns=KEYS)
    part = part.sort_values(KEYS + ["val_f1", "ratio_key"], ascending=[True, True, True, False, True])
    part = part.drop_duplicates(KEYS, keep="first")
    columns = KEYS + ["val_f1", "test_f1", "ratio"]
    part = part[columns].rename(
        columns={
            "val_f1": f"{prefix}_val_f1",
            "test_f1": f"{prefix}_test_f1",
            "ratio": f"{prefix}_r",
        }
    )
    return part


def main() -> None:
    executions = pd.read_csv(SOURCE)
    selected_source = pd.read_csv(SELECTED_SOURCE)
    executions = executions.reset_index(drop=True)
    executions["candidate_row_id"] = np.arange(len(executions), dtype=int)
    executions["ratio_key"] = executions["ratio"].fillna(-1.0)
    executions["candidate"] = [
        candidate_label(constructor, ratio)
        for constructor, ratio in zip(executions["constructor"], executions["ratio"])
    ]
    executions["eligible_for_gate_1pp"] = executions["constructor"].isin(ELIGIBLE_CONSTRUCTORS)
    executions["dataset"] = executions["graph_id"]
    executions["backbone"] = "gcn"
    executions["seed"] = executions["pipeline_seed"].astype(int)
    executions["split_id"] = executions["task_id"]

    run_ids = selected_source[KEYS + ["run_id"]].drop_duplicates(KEYS)
    assert not run_ids.duplicated(KEYS).any(), "run_id mapping is not unique"
    executions = executions.merge(run_ids, on=KEYS, how="left", validate="many_to_one")
    assert executions["run_id"].notna().all(), "missing run_id mapping"

    # Freeze the selector using validation-only columns. test_f1 is intentionally
    # absent from this frame and cannot influence ordering or the 1 p.p. gate.
    selector = executions.loc[executions["eligible_for_gate_1pp"], KEYS + [
        "candidate_row_id", "constructor", "candidate", "ratio", "ratio_key", "val_f1"
    ]].copy()
    selector["tie_rank"] = selector["constructor"].map(PREFERENCE)
    selector = selector.sort_values(
        KEYS + ["val_f1", "tie_rank", "ratio_key"],
        ascending=[True, True, True, False, True, True],
    )
    pre_gate = selector.drop_duplicates(KEYS, keep="first").copy()

    originals = selector[selector["constructor"].eq("original")][
        KEYS + ["candidate_row_id", "val_f1"]
    ].rename(columns={"candidate_row_id": "original_row_id", "val_f1": "original_val_f1"})
    assert not originals.duplicated(KEYS).any(), "original candidate is not unique"
    pre_gate = pre_gate.merge(originals, on=KEYS, how="left", validate="one_to_one")
    pre_gate["validation_advantage_pp"] = 100.0 * (pre_gate["val_f1"] - pre_gate["original_val_f1"])
    use_rewiring = (
        ~pre_gate["constructor"].eq("original")
        & pre_gate["validation_advantage_pp"].ge(MARGIN_PP)
    )
    pre_gate["selected_row_id"] = np.where(
        use_rewiring, pre_gate["candidate_row_id"], pre_gate["original_row_id"]
    ).astype(int)

    # Only after selected_row_id is frozen do we join test outcomes.
    outcome_columns = [
        "candidate_row_id", "candidate", "constructor", "ratio", "val_f1", "test_f1",
        "graph_id", "task_id", "pipeline_seed", "feature_regime", "task_seed",
        "dataset", "backbone", "seed", "split_id", "run_id", "family", "configuration_id",
        "param_n", "param_target_avg_degree", "param_latent_homophily",
        "param_corruption_fraction", "param_corruption_mode",
    ]
    outcomes = executions[outcome_columns].copy()
    chosen = pre_gate.merge(
        outcomes,
        left_on="selected_row_id",
        right_on="candidate_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_pre_gate", ""),
    )
    original_outcomes = outcomes[outcomes["constructor"].eq("original")][
        KEYS + ["test_f1"]
    ].rename(columns={"test_f1": "original_test_f1"})
    chosen = chosen.merge(original_outcomes, on=KEYS, how="left", validate="one_to_one")
    chosen["selected_test_f1"] = chosen["test_f1"]
    chosen["selected_candidate"] = chosen["candidate"]
    chosen["selected_r"] = np.where(
        chosen["selected_candidate"].isin(["delaunay", "vgae"]), chosen["ratio"], np.nan
    )
    chosen["validation_score_selected"] = chosen["val_f1"]
    chosen["validation_score_original"] = chosen["original_val_f1"]
    chosen["gain_f1"] = chosen["selected_test_f1"] - chosen["original_test_f1"]
    chosen["gain_pp"] = 100.0 * chosen["gain_f1"]
    chosen["pre_gate_best_candidate"] = chosen["candidate_pre_gate"]
    chosen["pre_gate_best_r"] = np.where(
        chosen["pre_gate_best_candidate"].eq("vgae"), chosen["ratio_pre_gate"], np.nan
    )
    chosen["pre_gate_best_val_f1"] = chosen["val_f1_pre_gate"]
    chosen["applied_margin_pp"] = MARGIN_PP

    # Requested candidate diagnostics. Each candidate-specific value is itself
    # chosen by validation only when multiple r values exist (VGAE).
    wide = chosen[[
        "dataset", "backbone", "seed", "split_id", "run_id", "graph_id", "task_id",
        "family", "configuration_id", "feature_regime", "task_seed", "pipeline_seed",
        "param_n", "param_target_avg_degree", "param_latent_homophily",
        "param_corruption_fraction", "param_corruption_mode",
        "original_test_f1", "selected_test_f1", "gain_f1", "gain_pp",
        "selected_candidate", "selected_r", "validation_score_selected",
        "validation_score_original", "validation_advantage_pp",
        "pre_gate_best_candidate", "pre_gate_best_r", "pre_gate_best_val_f1",
        "applied_margin_pp", "selected_row_id",
    ]].copy()

    diagnostics = [
        ("delaunay", "delaunay"),
        ("knn", "knn"),
        ("mutual_knn", "mutual_knn"),
        ("vgae", "vgae"),
        ("learned_hd_knn", "learned_hd_knn"),
        ("raw_hd_knn", "raw_hd_knn"),
        ("raw_umap_delaunay", "raw_umap_delaunay"),
        ("delaunay_union_original", "delaunay_union_original"),
    ]
    for candidate, prefix in diagnostics:
        wide = wide.merge(
            best_candidate_table(executions, candidate, prefix),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    # Put the exact requested columns first.
    first = [
        "dataset", "backbone", "seed",
        "original_test_f1", "selected_test_f1",
        "selected_candidate", "validation_score_selected", "validation_score_original",
        "split_id", "run_id",
        "delaunay_test_f1", "knn_test_f1", "mutual_knn_test_f1", "vgae_test_f1",
        "delaunay_val_f1", "knn_val_f1", "mutual_knn_val_f1", "vgae_val_f1",
        "selected_r",
    ]
    wide = wide[first + [column for column in wide.columns if column not in first]]
    wide = wide.sort_values(["dataset", "split_id", "seed"]).reset_index(drop=True)

    # Long format keeps every evaluated condition and flags eligibility and the
    # final gated selection. This is the most transparent audit representation.
    selected_ids = set(wide["selected_row_id"].astype(int))
    long_columns = [
        "dataset", "backbone", "seed", "split_id", "run_id", "graph_id", "task_id",
        "family", "configuration_id", "feature_regime", "task_seed", "pipeline_seed",
        "param_n", "param_target_avg_degree", "param_latent_homophily",
        "param_corruption_fraction", "param_corruption_mode", "candidate", "constructor",
        "stage", "ratio", "val_f1", "test_f1", "selection_metric", "selection_value",
        "eligible_for_gate_1pp", "candidate_row_id",
    ]
    long_df = executions[long_columns].copy()
    long_df["selected_by_gate_1pp"] = long_df["candidate_row_id"].isin(selected_ids)
    long_df = long_df.sort_values(["dataset", "split_id", "seed", "candidate", "ratio_key"] if "ratio_key" in long_df.columns else ["dataset", "split_id", "seed", "candidate", "ratio"])

    # Reproduce both per-execution and graph-level summaries. The published
    # +5.398 / 93.9% / 89.4% figures use graph-level means (n=180).
    graph = wide.groupby("graph_id", as_index=False).agg(gain_pp=("gain_pp", "mean"))
    rng = np.random.default_rng(20260918)
    graph_values = graph["gain_pp"].to_numpy()
    # The original summary reused one RNG across margins 0.0, 0.5, 1.0,
    # 2.0 and 3.0. Advance through the first two 2,000-bootstrap blocks so
    # the 1.0 p.p. interval reproduces the published file byte-for-number.
    dummy = np.zeros(len(graph_values), dtype=float)
    for _prior_margin in (0.0, 0.5):
        for _ in range(2000):
            rng.choice(dummy, size=len(dummy), replace=True)
    boot = np.array([
        np.mean(rng.choice(graph_values, size=len(graph_values), replace=True))
        for _ in range(2000)
    ])

    def wilcoxon_p(values: np.ndarray) -> float:
        try:
            return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            return float("nan")

    run_values = wide["gain_pp"].to_numpy()
    summary = pd.DataFrame([
        {
            "aggregation_level": "execution_seed",
            "n": len(run_values),
            "mean_gain_pp": float(np.mean(run_values)),
            "median_gain_pp": float(np.median(run_values)),
            "std_gain_pp": float(np.std(run_values, ddof=1)),
            "bootstrap_ci95_low_pp": np.nan,
            "bootstrap_ci95_high_pp": np.nan,
            "positive_fraction": float(np.mean(run_values > 0)),
            "gain_gt_0_5pp_fraction": float(np.mean(run_values > 0.5)),
            "wilcoxon_p_two_sided": wilcoxon_p(run_values),
            "note": "Rows are nested within graph_id; do not treat this p-value as the primary independent-unit inference.",
        },
        {
            "aggregation_level": "graph_mean_primary",
            "n": len(graph_values),
            "mean_gain_pp": float(np.mean(graph_values)),
            "median_gain_pp": float(np.median(graph_values)),
            "std_gain_pp": float(np.std(graph_values, ddof=1)),
            "bootstrap_ci95_low_pp": float(np.quantile(boot, 0.025)),
            "bootstrap_ci95_high_pp": float(np.quantile(boot, 0.975)),
            "positive_fraction": float(np.mean(graph_values > 0)),
            "gain_gt_0_5pp_fraction": float(np.mean(graph_values > 0.5)),
            "wilcoxon_p_two_sided": wilcoxon_p(graph_values),
            "note": "Primary unit used for the reported +5.398 p.p., 93.9%, and 89.4% figures.",
        },
    ])

    assert len(wide) == 1080, f"expected 1080 execution rows, found {len(wide)}"
    assert len(graph) == 180, f"expected 180 graph units, found {len(graph)}"
    assert wide[KEYS].duplicated().sum() == 0, "wide keys are not unique"
    assert np.isclose(graph_values.mean(), 5.397842191757679, atol=1e-12)
    assert np.isclose(np.mean(graph_values > 0), 0.9388888888888889, atol=1e-12)
    assert np.isclose(np.mean(graph_values > 0.5), 0.8944444444444445, atol=1e-12)
    original_mask = wide["selected_candidate"].eq("original")
    assert np.allclose(
        wide.loc[original_mask, "selected_test_f1"],
        wide.loc[original_mask, "original_test_f1"],
        equal_nan=False,
    ), "original selections do not reconcile"

    wide.to_csv(OUT_WIDE, index=False)
    long_df.to_csv(OUT_LONG, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"wide={OUT_WIDE} rows={len(wide)}")
    print(f"long={OUT_LONG} rows={len(long_df)}")
    print(f"summary={OUT_SUMMARY}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
