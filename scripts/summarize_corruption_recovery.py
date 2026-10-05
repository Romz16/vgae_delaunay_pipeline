
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text


ROOT = Path("outputs_revision/downloaded_results/corruption_recovery_pilot_20260918")


def main() -> None:
    graph = pd.read_csv(ROOT / "graph_level_outcomes.csv")
    selected = pd.read_csv(ROOT / "selected_execution_results.csv")
    executions = pd.read_csv(ROOT / "all_execution_results.csv")
    primary = pd.read_csv(ROOT / "recovery_primary_comparisons.csv")

    metadata = graph["generation_metadata"].map(json.loads)
    for field in (
        "clean_edge_homophily", "observed_edge_homophily", "clean_observed_edge_jaccard",
        "actual_changed_edge_fraction", "degree_sequence_preserved",
    ):
        graph[field] = metadata.map(lambda value: value.get(field, np.nan))

    original_runs = executions[executions.constructor.eq("original")].copy()
    original_graph = original_runs.groupby("graph_id", as_index=False).agg(
        baseline_val_f1=("val_f1", "mean"),
    )
    run_diagnostics = selected.groupby("graph_id", as_index=False).agg(
        baseline_test_f1=("baseline_test_f1", "mean"),
        learned_umap_trustworthiness=("learned_umap_trustworthiness", "mean"),
        raw_umap_trustworthiness=("raw_umap_trustworthiness", "mean"),
    )
    graph = graph.merge(run_diagnostics, on="graph_id", how="left").merge(original_graph, on="graph_id", how="left")
    graph["selected_gain_pp"] = 100 * graph["gain_selected_rewiring"]
    graph["geometry_gain_pp"] = 100 * graph["gain_geometry"]
    graph["vgae_increment_pp"] = 100 * graph["gain_vgae"]
    graph["success"] = (graph["selected_gain_pp"] > 0.5).astype(int)

    keys = ["graph_id", "task_id", "pipeline_seed"]
    baseline = executions[executions.constructor.eq("original")][keys + ["test_f1"]].rename(
        columns={"test_f1": "original_test_f1"}
    )
    compared = executions.merge(baseline, on=keys, how="left")
    compared["gain_pp"] = 100 * (compared.test_f1 - compared.original_test_f1)

    method_overall = primary.sort_values("mean_gain_pp", ascending=False).copy()
    method_overall.to_csv(ROOT / "analysis_method_overall.csv", index=False)

    proposed_strata = []
    for dimension in (
        "param_corruption_fraction", "param_corruption_mode", "param_latent_homophily",
        "param_n", "param_target_avg_degree",
    ):
        for level, part in graph.groupby(dimension, dropna=False):
            proposed_strata.append({
                "dimension": dimension, "level": level, "n_graphs": len(part),
                "mean_selected_gain_pp": part.selected_gain_pp.mean(),
                "median_selected_gain_pp": part.selected_gain_pp.median(),
                "success_fraction": part.success.mean(),
                "mean_geometry_gain_pp": part.geometry_gain_pp.mean(),
                "mean_vgae_increment_pp": part.vgae_increment_pp.mean(),
            })
    strata = pd.DataFrame(proposed_strata)
    strata.to_csv(ROOT / "analysis_proposed_method_strata.csv", index=False)

    observable = [
        "baseline_val_f1", "observed_edge_homophily", "density", "avg_degree", "degree_cv",
        "avg_clustering", "transitivity", "degree_assortativity", "lambda2_norm_laplacian",
        "algebraic_connectivity", "approx_avg_shortest_path", "approx_diameter",
        "effective_resistance_sample", "edge_connectivity_ratio", "spectral_sweep_conductance",
        "modularity", "num_components", "largest_cc_ratio", "num_isolates",
        "learned_umap_trustworthiness", "raw_umap_trustworthiness",
    ]
    metric_rows = []
    y = graph.success.to_numpy()
    for metric in observable:
        x = pd.to_numeric(graph[metric], errors="coerce")
        valid = x.notna() & graph.selected_gain_pp.notna()
        rho, pvalue = spearmanr(x[valid], graph.loc[valid, "selected_gain_pp"])
        auc_raw = roc_auc_score(y[valid], x[valid]) if len(np.unique(y[valid])) == 2 else np.nan
        if np.isnan(auc_raw):
            auc, direction = np.nan, "unavailable"
        elif auc_raw >= 0.5:
            auc, direction = auc_raw, "higher predicts success"
        else:
            auc, direction = 1 - auc_raw, "lower predicts success"
        metric_rows.append({
            "metric": metric, "n": int(valid.sum()), "spearman_rho_gain": rho,
            "spearman_p": pvalue, "single_metric_auc_oriented": auc, "direction": direction,
        })
    metric_table = pd.DataFrame(metric_rows).sort_values("single_metric_auc_oriented", ascending=False)
    metric_table.to_csv(ROOT / "analysis_suitability_metrics.csv", index=False)

    # Honest out-of-sample assessment: entire structural configurations stay
    # together, preventing near-duplicate graph instances from crossing folds.
    candidate = observable
    X = graph[candidate].apply(pd.to_numeric, errors="coerce")
    groups = graph.configuration_id
    splitter = GroupKFold(n_splits=5)
    probabilities = np.full(len(graph), np.nan)
    predictions = np.full(len(graph), -1)
    for train, test in splitter.split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=20260918),
        )
        model.fit(X.iloc[train], y[train])
        probabilities[test] = model.predict_proba(X.iloc[test])[:, 1]
        predictions[test] = (probabilities[test] >= 0.5).astype(int)
    multivariate_auc = roc_auc_score(y, probabilities)
    multivariate_balanced_accuracy = balanced_accuracy_score(y, predictions)

    interpretable = [
        "baseline_val_f1", "observed_edge_homophily", "lambda2_norm_laplacian", "density",
        "degree_cv", "avg_clustering", "effective_resistance_sample",
        "learned_umap_trustworthiness", "raw_umap_trustworthiness",
    ]
    tree_X = graph[interpretable].apply(pd.to_numeric, errors="coerce")
    tree_probabilities = np.full(len(graph), np.nan)
    tree_predictions = np.full(len(graph), -1)
    for train, test in splitter.split(tree_X, y, groups):
        tree_model = make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=2, min_samples_leaf=18, class_weight="balanced", random_state=20260918),
        )
        tree_model.fit(tree_X.iloc[train], y[train])
        tree_probabilities[test] = tree_model.predict_proba(tree_X.iloc[test])[:, 1]
        tree_predictions[test] = (tree_probabilities[test] >= 0.5).astype(int)
    tree_auc = roc_auc_score(y, tree_probabilities)
    tree_balanced_accuracy = balanced_accuracy_score(y, tree_predictions)
    final_tree = DecisionTreeClassifier(
        max_depth=2, min_samples_leaf=18, class_weight="balanced", random_state=20260918
    )
    imputer = SimpleImputer(strategy="median")
    final_tree.fit(imputer.fit_transform(tree_X), y)
    tree_rules = export_text(final_tree, feature_names=interpretable, decimals=4)

    # Task-level suitability is the deployable view: validation performance is
    # measured for each feature regime and all outcomes remain on the test set.
    task_keys = ["graph_id", "task_id", "pipeline_seed"]
    task_base = original_runs[task_keys + ["val_f1"]].rename(columns={"val_f1": "baseline_val_f1"})
    task = selected.merge(task_base, on=task_keys, how="left")
    task["gain_pp"] = 100 * task.gain_selected_rewiring
    task_units = task.groupby(
        ["graph_id", "configuration_id", "feature_regime"], as_index=False
    ).agg(
        baseline_val_f1=("baseline_val_f1", "mean"), gain_pp=("gain_pp", "mean"),
        learned_umap_trustworthiness=("learned_umap_trustworthiness", "mean"),
        raw_umap_trustworthiness=("raw_umap_trustworthiness", "mean"),
    )
    graph_features = graph.drop(columns=["baseline_val_f1", "learned_umap_trustworthiness", "raw_umap_trustworthiness"])
    task_units = task_units.merge(graph_features, on=["graph_id", "configuration_id"], how="left")
    task_units["success"] = (task_units.gain_pp > 0.5).astype(int)
    topology_features = [
        "density", "avg_degree", "degree_cv", "avg_clustering", "transitivity",
        "degree_assortativity", "lambda2_norm_laplacian", "algebraic_connectivity",
        "approx_avg_shortest_path", "approx_diameter", "effective_resistance_sample",
        "spectral_sweep_conductance", "modularity", "num_components", "largest_cc_ratio", "num_isolates",
    ]
    label_structure_features = topology_features + ["observed_edge_homophily"]
    operational_features = label_structure_features + [
        "baseline_val_f1", "learned_umap_trustworthiness", "raw_umap_trustworthiness"
    ]

    def group_cv(feature_names: list[str]) -> tuple[float, float]:
        tx = task_units[feature_names].apply(pd.to_numeric, errors="coerce")
        ty = task_units.success.to_numpy()
        tg = task_units.configuration_id
        probs = np.full(len(task_units), np.nan); preds = np.full(len(task_units), -1)
        for train, test in splitter.split(tx, ty, tg):
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=20260918),
            )
            model.fit(tx.iloc[train], ty[train])
            probs[test] = model.predict_proba(tx.iloc[test])[:, 1]
            preds[test] = (probs[test] >= 0.5).astype(int)
        return roc_auc_score(ty, probs), balanced_accuracy_score(ty, preds)

    baseline_only_auc, baseline_only_bacc = group_cv(["baseline_val_f1"])
    topology_auc, topology_bacc = group_cv(topology_features)
    label_structure_auc, label_structure_bacc = group_cv(label_structure_features)
    operational_auc, operational_bacc = group_cv(operational_features)

    bins = [-np.inf, 0.40, 0.55, 0.70, np.inf]
    labels = ["<=0.40", "0.40–0.55", "0.55–0.70", ">0.70"]
    task_units["baseline_val_band"] = pd.cut(task_units.baseline_val_f1, bins=bins, labels=labels)
    baseline_bands = task_units.groupby("baseline_val_band", observed=True, as_index=False).agg(
        n=("gain_pp", "size"), mean_gain_pp=("gain_pp", "mean"),
        median_gain_pp=("gain_pp", "median"), success_fraction=("success", "mean"),
    )
    baseline_bands.to_csv(ROOT / "analysis_baseline_validation_bands.csv", index=False)

    threshold_rows = []
    for threshold in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70):
        for relation, mask in (("at_or_below", task_units.baseline_val_f1 <= threshold),
                               ("above", task_units.baseline_val_f1 > threshold)):
            part = task_units[mask]
            threshold_rows.append({
                "threshold": threshold, "relation": relation, "n": len(part),
                "coverage": len(part) / len(task_units), "mean_gain_pp": part.gain_pp.mean(),
                "median_gain_pp": part.gain_pp.median(), "success_fraction": part.success.mean(),
            })
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(ROOT / "analysis_baseline_validation_thresholds.csv", index=False)

    # Leakage-safe operational selector. The original graph participates as a
    # candidate and rewiring is allowed only when validation improves by a
    # configurable margin. Test F1 is never used in the choice.
    operational_names = {
        "original", "raw_hd_knn", "raw_umap_delaunay", "learned_hd_knn",
        "delaunay", "delaunay_union_original",
    }
    candidates = executions[executions.constructor.isin(operational_names)].copy()
    candidates["ratio_key"] = candidates.ratio.fillna(-1.0)
    preference = {
        "original": 0, "raw_hd_knn": 1, "raw_umap_delaunay": 2, "learned_hd_knn": 3,
        "delaunay_union_original": 4, "delaunay": 5,
    }
    candidates["tie_rank"] = candidates.constructor.map(preference)
    original_scores = candidates[candidates.constructor.eq("original")][
        keys + ["val_f1", "test_f1"]
    ].rename(columns={"val_f1": "original_val_f1", "test_f1": "original_test_f1"})
    ranked = candidates.sort_values(
        keys + ["val_f1", "tie_rank", "ratio_key"],
        ascending=[True, True, True, False, True, True],
    )
    best = ranked.drop_duplicates(keys, keep="first").merge(original_scores, on=keys, how="left")
    best["validation_advantage"] = best.val_f1 - best.original_val_f1
    gate_rows = []
    choice_rows = []
    choice_performance_rows = []
    rng_gate = np.random.default_rng(20260918)
    for margin_pp in (0.0, 0.5, 1.0, 2.0, 3.0):
        use = (~best.constructor.eq("original")) & (100 * best.validation_advantage >= margin_pp)
        best["gated_constructor"] = np.where(use, best.constructor, "original")
        best["gated_ratio"] = np.where(use, best.ratio, np.nan)
        best["gated_test_f1"] = np.where(use, best.test_f1, best.original_test_f1)
        best["gated_gain_pp"] = 100 * (best.gated_test_f1 - best.original_test_f1)
        units = best.groupby("graph_id", as_index=False).agg(gain_pp=("gated_gain_pp", "mean"))
        values = units.gain_pp.to_numpy()
        boot = np.array([
            np.mean(rng_gate.choice(values, size=len(values), replace=True)) for _ in range(2000)
        ])
        gate_rows.append({
            "validation_margin_pp": margin_pp, "n_graphs": len(values),
            "rewiring_run_fraction": float(use.mean()), "mean_gain_pp": float(values.mean()),
            "median_gain_pp": float(np.median(values)), "ci95_low_pp": float(np.quantile(boot, 0.025)),
            "ci95_high_pp": float(np.quantile(boot, 0.975)),
            "positive_graph_fraction": float(np.mean(values > 0)),
            "success_graph_fraction": float(np.mean(values > 0.5)),
        })
        frequencies = best.groupby(["feature_regime", "gated_constructor"], as_index=False).size()
        frequencies["validation_margin_pp"] = margin_pp
        frequencies["fraction_within_regime"] = frequencies["size"] / frequencies.groupby("feature_regime")["size"].transform("sum")
        choice_rows.append(frequencies)
        performance = best.groupby(["feature_regime", "gated_constructor"], as_index=False).agg(
            n=("gated_gain_pp", "size"), mean_test_gain_pp=("gated_gain_pp", "mean"),
            median_test_gain_pp=("gated_gain_pp", "median"),
            positive_run_fraction=("gated_gain_pp", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
        performance["validation_margin_pp"] = margin_pp
        choice_performance_rows.append(performance)
    gate_table = pd.DataFrame(gate_rows)
    gate_table.to_csv(ROOT / "analysis_validation_gating.csv", index=False)
    pd.concat(choice_rows, ignore_index=True).to_csv(ROOT / "analysis_validation_gating_choices.csv", index=False)
    pd.concat(choice_performance_rows, ignore_index=True).to_csv(
        ROOT / "analysis_validation_gating_choice_performance.csv", index=False
    )

    best_metric = metric_table.iloc[0]
    result = {
        "graphs": int(len(graph)), "nested_runs": int(len(selected)),
        "gnn_conditions": int(len(executions)), "success_threshold_pp": 0.5,
        "selected_method_mean_gain_pp": float(graph.selected_gain_pp.mean()),
        "selected_method_median_gain_pp": float(graph.selected_gain_pp.median()),
        "selected_method_positive_fraction": float((graph.selected_gain_pp > 0).mean()),
        "selected_method_success_fraction": float(graph.success.mean()),
        "geometry_mean_gain_pp": float(graph.geometry_gain_pp.mean()),
        "vgae_increment_mean_pp": float(graph.vgae_increment_pp.mean()),
        "best_single_metric": str(best_metric.metric),
        "best_single_metric_auc_oriented": float(best_metric.single_metric_auc_oriented),
        "best_single_metric_direction": str(best_metric.direction),
        "multivariate_group_cv_auc": float(multivariate_auc),
        "multivariate_group_cv_balanced_accuracy": float(multivariate_balanced_accuracy),
        "tree_group_cv_auc": float(tree_auc),
        "tree_group_cv_balanced_accuracy": float(tree_balanced_accuracy),
        "tree_rules_full_sample_exploratory": tree_rules,
        "task_level_baseline_only_group_cv_auc": float(baseline_only_auc),
        "task_level_baseline_only_group_cv_balanced_accuracy": float(baseline_only_bacc),
        "task_level_topology_only_group_cv_auc": float(topology_auc),
        "task_level_topology_only_group_cv_balanced_accuracy": float(topology_bacc),
        "task_level_topology_plus_homophily_group_cv_auc": float(label_structure_auc),
        "task_level_topology_plus_homophily_group_cv_balanced_accuracy": float(label_structure_bacc),
        "task_level_operational_group_cv_auc": float(operational_auc),
        "task_level_operational_group_cv_balanced_accuracy": float(operational_bacc),
        "best_validation_gate_margin_pp_exploratory": float(gate_table.loc[gate_table.mean_gain_pp.idxmax(), "validation_margin_pp"]),
        "best_validation_gate_mean_gain_pp_exploratory": float(gate_table.mean_gain_pp.max()),
    }
    (ROOT / "analysis_key_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    top = metric_table.head(8)
    corruption = strata[strata.dimension.eq("param_corruption_fraction")]
    homophily = strata[strata.dimension.eq("param_latent_homophily")]
    lines = [
        "# Conclusões do experimento de corrupção–recuperação", "",
        f"Foram analisados {len(graph)} grafos independentes, {len(selected)} execuções aninhadas e {len(executions)} condições GNN.", "",
        "## Método proposto com seleção por validação", "",
        f"- Ganho médio: {graph.selected_gain_pp.mean():.3f} pp.",
        f"- Mediana: {graph.selected_gain_pp.median():.3f} pp.",
        f"- Fração positiva: {100*(graph.selected_gain_pp > 0).mean():.1f}%.",
        f"- Fração acima de 0,5 pp: {100*graph.success.mean():.1f}%.",
        f"- Delaunay sem refinamento: {graph.geometry_gain_pp.mean():.3f} pp.",
        f"- Incremento do refinamento VGAE: {graph.vgae_increment_pp.mean():.3f} pp.", "",
        "## Sensibilidade do método proposto", "",
        "### Corrupção", "",
    ]
    for row in corruption.itertuples():
        lines.append(f"- nível {row.level}: ganho {row.mean_selected_gain_pp:.3f} pp; sucesso {100*row.success_fraction:.1f}%.")
    lines += ["", "### Homofilia latente", ""]
    for row in homophily.itertuples():
        lines.append(f"- nível {row.level}: ganho {row.mean_selected_gain_pp:.3f} pp; sucesso {100*row.success_fraction:.1f}%.")
    lines += ["", "## Métricas candidatas", "", "| Métrica | Spearman com ganho | AUC orientada | Direção |", "|---|---:|---:|---|"]
    for row in top.itertuples():
        lines.append(f"| {row.metric} | {row.spearman_rho_gain:.3f} | {row.single_metric_auc_oriented:.3f} | {row.direction} |")
    lines += [
        "", "## Validação fora da amostra por configuração", "",
        f"- Modelo multivariado: AUC {multivariate_auc:.3f}; balanced accuracy {multivariate_balanced_accuracy:.3f}.",
        f"- Árvore rasa: AUC {tree_auc:.3f}; balanced accuracy {tree_balanced_accuracy:.3f}.",
        f"- Somente topologia, por tarefa: AUC {topology_auc:.3f}; balanced accuracy {topology_bacc:.3f}.",
        f"- Topologia + homofilia, por tarefa: AUC {label_structure_auc:.3f}; balanced accuracy {label_structure_bacc:.3f}.",
        f"- Somente F1 de validação do baseline, por tarefa: AUC {baseline_only_auc:.3f}; balanced accuracy {baseline_only_bacc:.3f}.",
        f"- Modelo operacional completo, por tarefa: AUC {operational_auc:.3f}; balanced accuracy {operational_bacc:.3f}.",
        "", "## Faixas do F1 de validação do baseline", "",
        "| Faixa | n | Ganho médio (pp) | Sucesso >0,5 pp |", "|---|---:|---:|---:|",
        *[f"| {row.baseline_val_band} | {row.n} | {row.mean_gain_pp:.3f} | {100*row.success_fraction:.1f}% |" for row in baseline_bands.itertuples()],
        "", "## Gate de validação entre construtores", "",
        "| Margem mínima na validação (pp) | Execuções com rewiring | Ganho médio no teste (pp) | IC95% |", "|---:|---:|---:|---:|",
        *[f"| {row.validation_margin_pp:.1f} | {100*row.rewiring_run_fraction:.1f}% | {row.mean_gain_pp:.3f} | [{row.ci95_low_pp:.3f}, {row.ci95_high_pp:.3f}] |" for row in gate_table.itertuples()],
        "", "A árvore abaixo é apenas descritiva no conjunto completo; não deve ser usada como regra confirmatória sem validação externa.", "", "```", tree_rules.rstrip(), "```", "",
    ]
    (ROOT / "RELATORIO_CONCLUSOES_RECOVERY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
