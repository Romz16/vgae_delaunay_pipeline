
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs_revision/downloaded_results")
MAIN = ROOT / "global_12datasets_corrected_20260824" / "all_runs_long.csv"
SOTA = ROOT / "global_protocol_matched_sota_controls_N30_lean" / "sota_all_runs.csv"
OUT = Path("outputs_revision/robustness_gate_analysis_20261002")
KEYS = ["dataset", "backbone", "seed"]
N_BOOT = 10_000
SEED = 20261002


def original_rows(main: pd.DataFrame) -> pd.DataFrame:
    original = main.loc[main["is_baseline"].astype(str).str.lower().eq("true"), KEYS + ["val_f1", "test_f1"]].copy()
    if original.duplicated(KEYS).any():
        raise ValueError("Original graph rows are not unique.")
    original["candidate"] = "original"
    original["ratio"] = -1.0
    return original


def choose_by_validation(rows: pd.DataFrame, method: str) -> pd.DataFrame:
    frame = rows.copy()
    frame["is_original"] = frame["candidate"].eq("original")
    frame = frame.sort_values(KEYS + ["val_f1", "is_original", "ratio"],
                              ascending=[True, True, True, False, False, True])
    selected = frame.drop_duplicates(KEYS, keep="first").copy()
    selected["method"] = method
    selected["fallback_to_original"] = selected["candidate"].eq("original")
    return selected


def bootstrap_difference(paired: pd.DataFrame, label: str) -> dict[str, object]:
    datasets = np.array(sorted(paired.dataset.unique()))
    by_dataset = {d: x.delta_pp.to_numpy(dtype=float) for d, x in paired.groupby("dataset")}
    rng = np.random.default_rng(SEED + sum(map(ord, label)))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled_datasets = rng.choice(datasets, len(datasets), replace=True)
        draws[i] = np.mean([
            np.mean(rng.choice(by_dataset[d], len(by_dataset[d]), replace=True))
            for d in sampled_datasets
        ])
    return {
        "comparison": label,
        "n_datasets": len(datasets),
        "n_paired_executions": len(paired),
        "mean_difference_pp": paired.delta_pp.mean(),
        "ci95_low_pp": np.quantile(draws, .025),
        "ci95_high_pp": np.quantile(draws, .975),
        "bootstrap_replicates": N_BOOT,
        "resampling": "datasets, then shared backbone/seed executions within each dataset",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = pd.read_csv(MAIN)
    sota = pd.read_csv(SOTA)
    # Baseline rows are shared by all candidates. Limit to the exact 30 common seeds.
    common = set(sota.seed.astype(int).unique())
    original = original_rows(main_rows)
    original = original.loc[original.seed.astype(int).isin(common)].copy()

    # Proposed method + original fallback.
    proposed_candidates = main_rows.loc[
        ~main_rows["is_baseline"].astype(str).str.lower().eq("true") & main_rows.seed.astype(int).isin(common),
        KEYS + ["condition", "ratio", "val_f1", "test_f1"],
    ].rename(columns={"condition": "candidate"})
    proposed_candidates["ratio"] = pd.to_numeric(proposed_candidates["ratio"], errors="raise")
    proposed = choose_by_validation(pd.concat([proposed_candidates, original], ignore_index=True), "Proposed")

    # SDRF's 0% snapshot has original topology, but it was trained separately
    # and its recorded metrics are not bit-identical to the shared baseline.
    # Remove that duplicate topology and append the same saved original row
    # used for Proposed and DiffWire, which makes the fallback exactly common.
    sdrf = sota.loc[sota.method.eq("SDRF")].copy()
    sdrf["ratio"] = pd.to_numeric(sdrf.candidate_ratio, errors="raise")
    sdrf = sdrf.loc[~np.isclose(sdrf["ratio"], 0.0), KEYS + ["ratio", "val_f1", "test_f1"]]
    sdrf["candidate"] = "SDRF"
    sdrf = choose_by_validation(pd.concat([sdrf, original], ignore_index=True), "SDRF")

    # DiffWire CT has one reweighted-graph candidate, so append the same original rows.
    diffwire = sota.loc[sota.method.eq("DiffWire CT"), KEYS + ["val_f1", "test_f1"]].copy()
    diffwire["candidate"] = "DiffWire CT"
    diffwire["ratio"] = 0.0
    diffwire = choose_by_validation(pd.concat([diffwire, original], ignore_index=True), "DiffWire CT")

    method_rows = pd.concat([proposed, sdrf, diffwire], ignore_index=True)
    original_for_gain = original[KEYS + ["test_f1"]].rename(columns={"test_f1": "original_test_f1"})
    method_rows = method_rows.merge(original_for_gain, on=KEYS, how="inner", validate="many_to_one")
    method_rows["gain_vs_original_pp"] = 100 * (method_rows.test_f1 - method_rows.original_test_f1)
    method_rows = method_rows.rename(columns={"val_f1": "selected_val_f1", "test_f1": "selected_test_f1", "ratio": "selected_ratio"})

    summaries = method_rows.groupby("method", as_index=False).agg(
        n=("seed", "count"),
        selected_test_f1_mean=("selected_test_f1", "mean"),
        mean_gain_vs_original_pp=("gain_vs_original_pp", "mean"),
        original_fallback_rate=("fallback_to_original", "mean"),
    )
    # Compare the proposed method to a comparator only on their common keys.
    comparisons = []
    proposed_for_merge = method_rows.loc[method_rows.method.eq("Proposed"), KEYS + ["selected_test_f1"]].rename(
        columns={"selected_test_f1": "proposed_test_f1"}
    )
    for baseline in ("SDRF", "DiffWire CT"):
        base = method_rows.loc[method_rows.method.eq(baseline), KEYS + ["selected_test_f1"]].rename(
            columns={"selected_test_f1": "baseline_test_f1"}
        )
        paired = proposed_for_merge.merge(base, on=KEYS, how="inner", validate="one_to_one")
        paired["delta_pp"] = 100 * (paired.proposed_test_f1 - paired.baseline_test_f1)
        comparisons.append(bootstrap_difference(paired, f"Proposed fallback minus {baseline} fallback"))
    comparisons = pd.DataFrame(comparisons)

    method_rows.sort_values(["method", *KEYS]).to_csv(OUT / "protocol_matched_fallback_selected_per_seed.csv", index=False)
    summaries.to_csv(OUT / "protocol_matched_fallback_method_summary.csv", index=False)
    comparisons.to_csv(OUT / "protocol_matched_fallback_bootstrap.csv", index=False)

    report = ["# Fallback equivalente para comparadores SOTA", "",
              "Seleção refeita exclusivamente por macro-F1 de validação. Cada método recebeu o candidato `original`; o teste foi usado somente depois da seleção.",
              "", "## Regra comum de fallback", "",
              "O snapshot SDRF de 0% possui a topologia original, mas foi treinado em uma invocação separada e não reproduz bit a bit os valores do baseline salvo. Para evitar que qualquer método receba uma realização diferente do original, ele foi removido da grade SDRF e o mesmo baseline original salvo foi adicionado explicitamente a Proposed, SDRF e DiffWire CT.",
              "", "## Resultados"]
    for row in summaries.itertuples(index=False):
        report.append(f"- {row.method}: ganho médio contra o original = {row.mean_gain_vs_original_pp:.3f} p.p.; fallback para original = {100*row.original_fallback_rate:.1f}% (n={int(row.n)}).")
    report += ["", "## Comparações pareadas com fallback em ambos os lados"]
    for row in comparisons.itertuples(index=False):
        report.append(f"- {row.comparison}: {row.mean_difference_pp:.3f} p.p. (IC 95% [{row.ci95_low_pp:.3f}, {row.ci95_high_pp:.3f}], n={int(row.n_paired_executions)}).")
    (OUT / "PROTOCOL_MATCHED_FALLBACK_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summaries.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
