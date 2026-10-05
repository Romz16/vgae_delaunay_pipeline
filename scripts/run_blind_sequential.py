#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "lambda2_norm_laplacian", "algebraic_connectivity", "density", "avg_degree",
    "degree_cv", "avg_clustering", "transitivity", "degree_assortativity",
    "approx_avg_shortest_path", "approx_diameter", "num_components", "largest_cc_ratio",
    "num_isolates", "effective_resistance_sample", "edge_connectivity_ratio",
    "spectral_sweep_conductance", "modularity",
]
THRESHOLD_PP = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "reveal", "finalize"))
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument("--base-outcomes", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--target-output-dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--wave")
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--score-files", type=Path, nargs="*")
    parser.add_argument("--report-output", type=Path)
    return parser.parse_args()


def minimal_knowledge(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["graph_id", "family", "configuration_id", *FEATURES]].copy()
    result["gain_pp"] = 100 * frame["gain_selected_rewiring"] if "gain_selected_rewiring" in frame else frame["gain_pp"]
    result["success"] = (result.gain_pp > THRESHOLD_PP).astype(int)
    return result


def load_knowledge(path: Path, base: Path | None) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if base is None:
        raise FileNotFoundError("Knowledge does not exist and --base-outcomes was not supplied.")
    knowledge = minimal_knowledge(pd.read_csv(base))
    path.parent.mkdir(parents=True, exist_ok=True)
    knowledge.to_csv(path, index=False)
    return knowledge


def models(seed: int = 20260918):
    linear = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000, random_state=seed),
    )
    trees = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=600, max_depth=5, min_samples_leaf=8, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
    )
    return linear, trees


def ensemble_fit_predict(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    x_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    x_target = target[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = train.success.to_numpy(int)
    linear, trees = models()
    linear.fit(x_train, y); trees.fit(x_train, y)
    return 0.5 * (linear.predict_proba(x_target)[:, 1] + trees.predict_proba(x_target)[:, 1])


def cv_diagnostics(knowledge: pd.DataFrame) -> dict[str, float]:
    x = knowledge[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = knowledge.success.to_numpy(int); groups = knowledge.configuration_id
    splits = min(5, groups.nunique())
    probabilities = np.full(len(knowledge), np.nan)
    for train, test in GroupKFold(n_splits=splits).split(x, y, groups):
        probabilities[test] = ensemble_fit_predict(knowledge.iloc[train], knowledge.iloc[test])
    predicted = (probabilities >= 0.5).astype(int)
    return classification_metrics(y, predicted, probabilities)


def classification_metrics(y, predicted, probability) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def predict(args) -> None:
    required = (args.catalog, args.predictions, args.target_output_dir, args.wave)
    if any(value is None for value in required):
        raise ValueError("predict requires --catalog, --predictions, --target-output-dir, and --wave")
    outcome = args.target_output_dir / "graph_level_outcomes.csv"
    if outcome.exists():
        raise RuntimeError(f"Blindness violation: outcomes already exist at {outcome}")
    knowledge = load_knowledge(args.knowledge, args.base_outcomes)
    catalog = pd.read_csv(args.catalog)
    overlap = set(knowledge.graph_id) & set(catalog.graph_id)
    if overlap:
        raise RuntimeError(f"Target contains {len(overlap)} graph ids already present in knowledge.")
    probability = ensemble_fit_predict(knowledge, catalog)
    output = catalog[["graph_id", "family", "configuration_id", *FEATURES]].copy()
    output.insert(0, "wave", args.wave)
    output["predicted_probability"] = probability
    output["predicted_success"] = (probability >= 0.5).astype(int)
    output["knowledge_graphs_before_prediction"] = len(knowledge)
    output["prediction_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.predictions, index=False)
    digest = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    diagnostics = {
        "wave": args.wave, "prediction_file_sha256": digest,
        "features": FEATURES, "success_definition": f"mean selected gain > {THRESHOLD_PP} pp",
        "knowledge_class_balance": knowledge.success.value_counts().sort_index().to_dict(),
        "knowledge_group_cv": cv_diagnostics(knowledge),
    }
    args.predictions.with_suffix(".metadata.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


def reveal(args) -> None:
    required = (args.predictions, args.target_output_dir, args.score_output, args.wave)
    if any(value is None for value in required):
        raise ValueError("reveal requires --predictions, --target-output-dir, --score-output, and --wave")
    predictions = pd.read_csv(args.predictions)
    outcome_path = args.target_output_dir / "graph_level_outcomes.csv"
    outcomes = pd.read_csv(outcome_path)
    truth = minimal_knowledge(outcomes)[["graph_id", "gain_pp", "success"]]
    scored = predictions.merge(truth, on="graph_id", how="inner", validate="one_to_one")
    if len(scored) != len(predictions):
        raise RuntimeError(f"Only {len(scored)}/{len(predictions)} blind predictions received outcomes.")
    metrics = classification_metrics(
        scored.success.to_numpy(int), scored.predicted_success.to_numpy(int), scored.predicted_probability.to_numpy(float)
    )
    metrics.update({
        "wave": args.wave,
        "prediction_file_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "reveal_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.score_output, index=False)
    args.score_output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    knowledge = load_knowledge(args.knowledge, args.base_outcomes)
    additions = minimal_knowledge(outcomes)
    combined = pd.concat([knowledge, additions], ignore_index=True)
    combined = combined.drop_duplicates("graph_id", keep="first")
    combined.to_csv(args.knowledge, index=False)
    print(json.dumps(metrics, indent=2))


def finalize(args) -> None:
    if not args.score_files or args.report_output is None:
        raise ValueError("finalize requires --score-files and --report-output")
    frames = [pd.read_csv(path) for path in args.score_files]
    scored = pd.concat(frames, ignore_index=True)
    metrics = classification_metrics(
        scored.success.to_numpy(int), scored.predicted_success.to_numpy(int), scored.predicted_probability.to_numpy(float)
    )
    wave_metrics = []
    for wave, part in scored.groupby("wave", sort=False):
        row = classification_metrics(
            part.success.to_numpy(int), part.predicted_success.to_numpy(int), part.predicted_probability.to_numpy(float)
        )
        row["wave"] = wave; wave_metrics.append(row)
    knowledge = load_knowledge(args.knowledge, args.base_outcomes)
    x = knowledge[FEATURES].apply(pd.to_numeric, errors="coerce"); y = knowledge.success.to_numpy(int)
    linear, trees = models(); linear.fit(x, y); trees.fit(x, y)
    linear_coef = np.abs(linear.named_steps["logisticregression"].coef_[0])
    tree_importance = trees.named_steps["extratreesclassifier"].feature_importances_
    importance = pd.DataFrame({
        "feature": FEATURES,
        "linear_abs_standardized_coefficient": linear_coef,
        "extra_trees_importance": tree_importance,
    })
    importance["mean_normalized_importance"] = 0.5 * (
        importance.linear_abs_standardized_coefficient / max(importance.linear_abs_standardized_coefficient.sum(), 1e-12)
        + importance.extra_trees_importance / max(importance.extra_trees_importance.sum(), 1e-12)
    )
    importance = importance.sort_values("mean_normalized_importance", ascending=False)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.report_output.with_name("blind_feature_importance.csv"), index=False)
    pd.DataFrame(wave_metrics).to_csv(args.report_output.with_name("blind_accuracy_by_wave.csv"), index=False)
    scored.to_csv(args.report_output.with_name("blind_all_frozen_predictions_scored.csv"), index=False)
    lines = [
        "# Validação cega sequencial", "",
        "As previsões de cada onda foram congeladas antes da execução do rewiring. Somente resultados verdadeiros foram adicionados ao conhecimento.", "",
        f"- Grafos cegos: {metrics['n']}", f"- Acurácia: {metrics['accuracy']:.3f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.3f}", f"- ROC AUC: {metrics['roc_auc']:.3f}",
        f"- Precisão: {metrics['precision']:.3f}", f"- Sensibilidade: {metrics['recall_sensitivity']:.3f}",
        f"- Especificidade: {metrics['specificity']:.3f}", f"- Brier score: {metrics['brier_score']:.3f}", "",
        "## Evolução por onda", "", "| Onda | n | Acurácia | Balanced accuracy | AUC |", "|---|---:|---:|---:|---:|",
    ]
    for row in wave_metrics:
        lines.append(f"| {row['wave']} | {row['n']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['roc_auc']:.3f} |")
    lines += ["", "## Informações do grafo mais utilizadas pelo modelo final", ""]
    for row in importance.head(10).itertuples():
        lines.append(f"- {row.feature}: importância combinada {row.mean_normalized_importance:.3f}")
    lines += ["", "A importância final é descritiva. A validade preditiva é medida apenas pelas previsões congeladas das ondas cegas.", ""]
    args.report_output.write_text("\n".join(lines), encoding="utf-8")
    args.report_output.with_suffix(".metrics.json").write_text(json.dumps({"overall": metrics, "waves": wave_metrics}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    args = parse_args()
    if args.phase == "predict": predict(args)
    elif args.phase == "reveal": reveal(args)
    else: finalize(args)


if __name__ == "__main__":
    main()
