"""Post-hoc robustness analyses from saved per-seed experiment outputs.

The script does not retrain models or use test values for selection.  It
reconstructs each validation-based choice using only ``val_f1``, then joins the
corresponding held-out ``test_f1`` to estimate the outcome.  It produces:

1. a dataset/seed hierarchical bootstrap for the primary 12-dataset protocol;
2. leave-one-dataset-out and leave-all-Airports-out sensitivity analyses; and
3. selection-regret diagnostics for the explicit 1 p.p. validation gate.

Run from the repository root:
    python scripts/analyze_robustness_and_gate_quality.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs_revision")
MAIN_SOURCE = ROOT / "downloaded_results" / "global_12datasets_corrected_20260824" / "all_runs_long.csv"
GATE_WIDE = ROOT / "downloaded_results" / "corruption_recovery_pilot_20260918" / "validation_gate_1pp_per_seed_wide.csv"
GATE_LONG = ROOT / "downloaded_results" / "corruption_recovery_pilot_20260918" / "validation_gate_1pp_candidates_long.csv"
SOTA_SOURCE = ROOT / "downloaded_results" / "global_protocol_matched_sota_controls_N30_lean" / "sota_all_runs.csv"
OUT = ROOT / "robustness_gate_analysis_20261002"

N_BOOTSTRAP = 10_000
RNG_SEED = 20261002
KEYS_MAIN = ["dataset", "backbone", "seed"]
KEYS_GATE = ["graph_id", "task_id", "pipeline_seed"]


def validation_select_rewiring(main: pd.DataFrame) -> pd.DataFrame:
    """Freeze the best rewired ratio by validation F1, then attach baseline.

    This reproduces the primary protocol's comparison: original graph versus
    the best *rewired* candidate.  The original graph is not included in this
    selector because the legacy 12-dataset protocol selected the rewiring rate
    among rewired candidates; the later 1 p.p. gate is analysed separately.
    """
    baseline = main.loc[main["is_baseline"].astype(str).str.lower().eq("true")].copy()
    rewired = main.loc[~main["is_baseline"].astype(str).str.lower().eq("true")].copy()
    if baseline.duplicated(KEYS_MAIN).any() or not set(KEYS_MAIN).issubset(rewired):
        raise ValueError("Primary rows are not uniquely keyed by dataset/backbone/seed.")

    # Stable sort makes ties reproducible, using lower rewiring ratio as tie-break.
    rewired["ratio_order"] = pd.to_numeric(rewired["ratio"], errors="raise")
    selected = (
        rewired.sort_values(KEYS_MAIN + ["val_f1", "ratio_order"], ascending=[True, True, True, False, True])
        .drop_duplicates(KEYS_MAIN, keep="first")
        .rename(columns={"test_f1": "selected_test_f1", "val_f1": "selected_val_f1", "condition": "selected_condition"})
    )
    base = baseline[KEYS_MAIN + ["test_f1", "val_f1"]].rename(
        columns={"test_f1": "original_test_f1", "val_f1": "original_val_f1"}
    )
    selected = selected.merge(base, on=KEYS_MAIN, how="inner", validate="one_to_one")
    selected["gain_pp"] = 100.0 * (selected["selected_test_f1"] - selected["original_test_f1"])
    return selected


def hierarchical_bootstrap_gain(selected: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Two-stage bootstrap: datasets, then paired backbone/seed executions."""
    datasets = np.array(sorted(selected["dataset"].unique()))
    by_dataset = {name: group["gain_pp"].to_numpy(dtype=float) for name, group in selected.groupby("dataset")}
    rng = np.random.default_rng(RNG_SEED)
    samples = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        drawn_datasets = rng.choice(datasets, size=len(datasets), replace=True)
        dataset_means = []
        for dataset in drawn_datasets:
            values = by_dataset[dataset]
            dataset_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples[i] = float(np.mean(dataset_means))

    result = pd.DataFrame([{
        "comparison": "validation-selected rewiring minus original graph",
        "metric": "test macro-F1 gain (percentage points)",
        "n_datasets": len(datasets),
        "n_paired_executions": len(selected),
        "point_estimate_mean_pp": selected["gain_pp"].mean(),
        "bootstrap_ci95_low_pp": np.quantile(samples, 0.025),
        "bootstrap_ci95_high_pp": np.quantile(samples, 0.975),
        "bootstrap_replicates": N_BOOTSTRAP,
        "resampling": "datasets with replacement; then paired backbone/seed executions within each drawn dataset",
        "random_seed": RNG_SEED,
    }])
    return result, samples


def hierarchical_bootstrap_pairwise_difference(
    paired: pd.DataFrame, comparison: str, n_bootstrap: int = N_BOOTSTRAP
) -> pd.DataFrame:
    """Hierarchical CI for a protocol-matched method-vs-method difference."""
    datasets = np.array(sorted(paired["dataset"].unique()))
    by_dataset = {name: group["difference_pp"].to_numpy(dtype=float) for name, group in paired.groupby("dataset")}
    # A different fixed stream per comparison leaves results reproducible without
    # making results depend on the ordering of the methods in the input file.
    rng = np.random.default_rng(RNG_SEED + sum(map(ord, comparison)))
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        drawn = rng.choice(datasets, size=len(datasets), replace=True)
        samples[i] = np.mean([
            np.mean(rng.choice(by_dataset[name], size=len(by_dataset[name]), replace=True))
            for name in drawn
        ])
    return pd.DataFrame([{
        "comparison": comparison,
        "metric": "proposed selected test macro-F1 minus baseline selected test macro-F1 (percentage points)",
        "n_datasets": len(datasets),
        "n_paired_executions": len(paired),
        "point_estimate_mean_pp": paired["difference_pp"].mean(),
        "bootstrap_ci95_low_pp": np.quantile(samples, 0.025),
        "bootstrap_ci95_high_pp": np.quantile(samples, 0.975),
        "bootstrap_replicates": n_bootstrap,
        "resampling": "datasets with replacement; then shared backbone/seed executions within each drawn dataset",
    }])


def protocol_matched_sota_bootstraps(main_selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare validation-selected proposed rows to validation-selected SOTA rows.

    The SOTA bundle contains 30 common seeds. SDRF has all three backbones;
    DiffWire CT is available only for GCN. Each method's candidate ratio is
    selected by its own validation F1 before test values are compared.
    """
    sota = pd.read_csv(SOTA_SOURCE)
    sota["ratio_order"] = pd.to_numeric(sota["candidate_ratio"], errors="raise")
    selected_sota = (
        sota.sort_values(["dataset", "method", "backbone", "seed", "val_f1", "ratio_order"],
                         ascending=[True, True, True, True, False, True])
        .drop_duplicates(["dataset", "method", "backbone", "seed"], keep="first")
        .rename(columns={"test_f1": "baseline_selected_test_f1", "method": "baseline_method"})
    )
    proposed = main_selected[KEYS_MAIN + ["selected_test_f1"]].copy()
    paired_rows: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for method in sorted(selected_sota["baseline_method"].unique()):
        comparator = selected_sota.loc[selected_sota["baseline_method"].eq(method)]
        paired = proposed.merge(
            comparator[KEYS_MAIN + ["baseline_method", "baseline_selected_test_f1"]],
            on=KEYS_MAIN, how="inner", validate="one_to_one"
        )
        paired["difference_pp"] = 100.0 * (paired["selected_test_f1"] - paired["baseline_selected_test_f1"])
        expected = 360 if method == "DiffWire CT" else 1080
        if len(paired) != expected:
            raise ValueError(f"Expected {expected} matched rows versus {method}, got {len(paired)}.")
        summaries.append(hierarchical_bootstrap_pairwise_difference(paired, f"proposed minus {method}"))
        paired_rows.append(paired)
    return pd.concat(summaries, ignore_index=True), pd.concat(paired_rows, ignore_index=True)


def leave_out(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    all_datasets = set(selected["dataset"].unique())
    for dataset in sorted(all_datasets):
        retained = selected.loc[selected["dataset"].ne(dataset)]
        rows.append({
            "analysis": "leave_one_dataset_out",
            "removed": dataset,
            "n_datasets_retained": retained["dataset"].nunique(),
            "n_paired_executions": len(retained),
            "mean_gain_pp": retained["gain_pp"].mean(),
            "median_gain_pp": retained["gain_pp"].median(),
            "positive_execution_fraction": (retained["gain_pp"] > 0).mean(),
        })
    airport_names = {"Airports-Brazil", "Airports-Europe", "Airports-USA"}
    if airport_names.issubset(all_datasets):
        retained = selected.loc[~selected["dataset"].isin(airport_names)]
        rows.append({
            "analysis": "leave_airports_group_out",
            "removed": "Airports-Brazil, Airports-Europe, Airports-USA",
            "n_datasets_retained": retained["dataset"].nunique(),
            "n_paired_executions": len(retained),
            "mean_gain_pp": retained["gain_pp"].mean(),
            "median_gain_pp": retained["gain_pp"].median(),
            "positive_execution_fraction": (retained["gain_pp"] > 0).mean(),
        })
    return pd.DataFrame(rows)


def gate_regret(wide: pd.DataFrame, long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure the test-performance opportunity cost of the frozen 1 p.p. gate."""
    eligible = long.loc[long["eligible_for_gate_1pp"].astype(str).str.lower().eq("true")].copy()
    eligible["ratio_order"] = pd.to_numeric(eligible["ratio"], errors="coerce").fillna(-1.0)
    # Test oracle is for diagnosis only: it never feeds the gate selection.
    oracle = (
        eligible.sort_values(KEYS_GATE + ["test_f1", "ratio_order", "candidate"], ascending=[True, True, True, False, True, True])
        .drop_duplicates(KEYS_GATE, keep="first")
        [KEYS_GATE + ["candidate", "ratio", "test_f1"]]
        .rename(columns={"candidate": "oracle_candidate", "ratio": "oracle_ratio", "test_f1": "oracle_test_f1"})
    )
    selected = wide.merge(oracle, on=KEYS_GATE, how="inner", validate="one_to_one")
    selected["selection_regret_pp"] = 100.0 * (selected["oracle_test_f1"] - selected["selected_test_f1"])
    selected["oracle_hit"] = np.isclose(selected["selection_regret_pp"], 0.0, atol=1e-10)
    selected["original_kept"] = selected["selected_candidate"].eq("original")

    summary = pd.DataFrame([
        {
            "scope": "all gate decisions",
            "n": len(selected),
            "mean_regret_pp": selected["selection_regret_pp"].mean(),
            "median_regret_pp": selected["selection_regret_pp"].median(),
            "p90_regret_pp": selected["selection_regret_pp"].quantile(0.90),
            "oracle_hit_rate": selected["oracle_hit"].mean(),
            "mean_selected_gain_vs_original_pp": selected["gain_pp"].mean(),
            "original_kept_rate": selected["original_kept"].mean(),
        },
        *[
            {
                "scope": label,
                "n": len(part),
                "mean_regret_pp": part["selection_regret_pp"].mean(),
                "median_regret_pp": part["selection_regret_pp"].median(),
                "p90_regret_pp": part["selection_regret_pp"].quantile(0.90),
                "oracle_hit_rate": part["oracle_hit"].mean(),
                "mean_selected_gain_vs_original_pp": part["gain_pp"].mean(),
                "original_kept_rate": part["original_kept"].mean(),
            }
            for label, part in [("gate kept original", selected.loc[selected["original_kept"]]),
                                ("gate adopted rewiring", selected.loc[~selected["original_kept"]])]
        ],
    ])
    detail = selected[[
        "dataset", "backbone", "seed", "graph_id", "task_id", "feature_regime",
        "param_corruption_fraction", "selected_candidate", "selected_r", "selected_test_f1",
        "original_test_f1", "gain_pp", "oracle_candidate", "oracle_ratio", "oracle_test_f1",
        "selection_regret_pp", "oracle_hit", "original_kept",
    ]].copy()
    return summary, detail


def write_markdown(bootstrap: pd.DataFrame, lodo: pd.DataFrame, regret: pd.DataFrame, sota_bootstrap: pd.DataFrame) -> None:
    b = bootstrap.iloc[0]
    airport = lodo.loc[lodo["analysis"].eq("leave_airports_group_out")].iloc[0]
    low, high = lodo.loc[lodo["analysis"].eq("leave_one_dataset_out"), "mean_gain_pp"].agg(["min", "max"])
    r = regret.loc[regret["scope"].eq("all gate decisions")].iloc[0]
    sota_lines = "\n".join(
        f"- {row.comparison}: {row.point_estimate_mean_pp:.3f} p.p. "
        f"(IC 95% [{row.bootstrap_ci95_low_pp:.3f}, {row.bootstrap_ci95_high_pp:.3f}]; n={int(row.n_paired_executions)})."
        for row in sota_bootstrap.itertuples(index=False)
    )
    report = f"""# Robustez e qualidade do gate

## Dados e regras

- Protocolo principal: 12 datasets, 3 backbones e 100 seeds por par dataset/backbone (3.600 comparações pareadas). A escolha do rewiring é reconstruída exclusivamente por `val_f1`; o teste só entra após congelar a escolha.
- Bootstrap hierárquico: reamostra primeiro os datasets e, dentro de cada dataset sorteado, as execuções pareadas backbone/seed. O efeito é ganho de macro-F1 no teste em pontos percentuais, rewiring selecionado por validação menos grafo original.
- Gate: avaliação separada do gate explícito de margem de 1,0 p.p. no experimento de recuperação por corrupção (1.080 decisões). O oráculo usa o melhor teste **somente para diagnóstico** e não é parte da seleção real.

## Bootstrap hierárquico

Ganho médio: **{b['point_estimate_mean_pp']:.3f} p.p.**. IC 95% hierárquico: **[{b['bootstrap_ci95_low_pp']:.3f}, {b['bootstrap_ci95_high_pp']:.3f}] p.p.**.

## Controles SOTA protocol-matched (30 seeds comuns)

{sota_lines}

## Leave-one-dataset-out

Ao retirar um dataset por vez, o ganho médio variou de **{low:.3f}** a **{high:.3f} p.p.**. Sem os três Airports simultaneamente, o ganho médio foi **{airport['mean_gain_pp']:.3f} p.p.** em {int(airport['n_datasets_retained'])} datasets restantes.

## Selection regret do gate de 1,0 p.p.

O regret médio foi **{r['mean_regret_pp']:.3f} p.p.**; a mediana foi **{r['median_regret_pp']:.3f} p.p.**; e o gate escolheu exatamente o mesmo candidato do melhor teste em **{100*r['oracle_hit_rate']:.1f}%** das decisões. O gate preservou o grafo original em **{100*r['original_kept_rate']:.1f}%** das decisões. O ganho médio efetivamente entregue pelo gate contra o original foi **{r['mean_selected_gain_vs_original_pp']:.3f} p.p.**.

## Interpretação

O bootstrap e o leave-one-dataset-out avaliam a robustez do efeito principal. O regret mede o custo de decidir com validação, em vez de conhecer antecipadamente o teste. Ele deve ser reportado como diagnóstico de seleção, sem afirmar que o oráculo é um resultado alcançável.
"""
    (OUT / "ROBUSTNESS_AND_GATE_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = pd.read_csv(MAIN_SOURCE)
    selected = validation_select_rewiring(main_rows)
    if len(selected) != 3600:
        raise ValueError(f"Expected 3,600 primary paired executions, got {len(selected)}.")
    bootstrap, _ = hierarchical_bootstrap_gain(selected)
    lodo = leave_out(selected)

    gate_wide = pd.read_csv(GATE_WIDE)
    gate_long = pd.read_csv(GATE_LONG)
    regret, regret_detail = gate_regret(gate_wide, gate_long)
    if len(regret_detail) != 1080:
        raise ValueError(f"Expected 1,080 gate decisions, got {len(regret_detail)}.")

    selected.to_csv(OUT / "primary_validation_selected_per_seed.csv", index=False)
    bootstrap.to_csv(OUT / "hierarchical_bootstrap_summary.csv", index=False)
    sota_bootstrap, sota_paired = protocol_matched_sota_bootstraps(selected)
    sota_bootstrap.to_csv(OUT / "hierarchical_bootstrap_protocol_matched_sota.csv", index=False)
    sota_paired.to_csv(OUT / "protocol_matched_sota_pairs_per_seed.csv", index=False)
    lodo.to_csv(OUT / "leave_one_dataset_out.csv", index=False)
    regret.to_csv(OUT / "selection_regret_gate_summary.csv", index=False)
    regret_detail.to_csv(OUT / "selection_regret_gate_per_seed.csv", index=False)
    write_markdown(bootstrap, lodo, regret, sota_bootstrap)
    print(bootstrap.to_string(index=False))
    print(sota_bootstrap.to_string(index=False))
    print(lodo.to_string(index=False))
    print(regret.to_string(index=False))
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
