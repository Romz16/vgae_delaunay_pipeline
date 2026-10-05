from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

base_path = Path("outputs_revision/downloaded_results/corruption_recovery_pilot_20260918/graph_level_outcomes.csv")
blind_path = Path("outputs_revision/downloaded_results/blind_sequential_20260919/blind_all_frozen_predictions_scored.csv")
output_path = blind_path.parent / "blind_single_metric_indicators.csv"

features = [
    "lambda2_norm_laplacian", "algebraic_connectivity", "density", "avg_degree", "degree_cv",
    "avg_clustering", "transitivity", "degree_assortativity", "approx_avg_shortest_path",
    "approx_diameter", "num_components", "largest_cc_ratio", "num_isolates",
    "effective_resistance_sample", "edge_connectivity_ratio", "spectral_sweep_conductance", "modularity",
]
train = pd.read_csv(base_path)
blind = pd.read_csv(blind_path)
y_train = (100 * train.gain_selected_rewiring > 0.5).astype(int).to_numpy()
y_blind = blind.success.astype(int).to_numpy()
rng = np.random.default_rng(20260919)
rows = []


def pairwise_auc(positive_scores, negative_scores):
    differences = np.asarray(positive_scores)[:, None] - np.asarray(negative_scores)[None, :]
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


for feature in features:
    x_train = pd.to_numeric(train[feature], errors="coerce")
    median = float(x_train.median())
    x_train = x_train.fillna(median).to_numpy(float)
    x_blind = pd.to_numeric(blind[feature], errors="coerce").fillna(median).to_numpy(float)
    raw_train_auc = roc_auc_score(y_train, x_train)
    direction = 1.0 if raw_train_auc >= 0.5 else -1.0
    train_score = direction * x_train
    blind_score = direction * x_blind
    fpr, tpr, thresholds = roc_curve(y_train, train_score)
    threshold = float(thresholds[np.argmax(tpr - fpr)])
    predicted = (blind_score >= threshold).astype(int)
    tn = int(np.sum((y_blind == 0) & (predicted == 0)))
    fp = int(np.sum((y_blind == 0) & (predicted == 1)))
    fn = int(np.sum((y_blind == 1) & (predicted == 0)))
    tp = int(np.sum((y_blind == 1) & (predicted == 1)))
    blind_auc = float(roc_auc_score(y_blind, blind_score))
    boot = []
    positive = np.flatnonzero(y_blind == 1); negative = np.flatnonzero(y_blind == 0)
    for _ in range(3000):
        pos_scores = blind_score[rng.choice(positive, len(positive), replace=True)]
        neg_scores = blind_score[rng.choice(negative, len(negative), replace=True)]
        boot.append(pairwise_auc(pos_scores, neg_scores))
    rows.append({
        "feature": feature,
        "direction_learned_on_old_graphs": "higher" if direction > 0 else "lower",
        "threshold_original_scale": threshold * direction,
        "train_oriented_auc": max(raw_train_auc, 1 - raw_train_auc),
        "blind_oriented_auc": blind_auc,
        "blind_auc_ci95_low": float(np.quantile(boot, 0.025)),
        "blind_auc_ci95_high": float(np.quantile(boot, 0.975)),
        "blind_balanced_accuracy": balanced_accuracy_score(y_blind, predicted),
        "blind_precision": precision_score(y_blind, predicted, zero_division=0),
        "blind_recall": recall_score(y_blind, predicted, zero_division=0),
        "blind_specificity": tn / max(tn + fp, 1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "blind_positive_median": float(np.median(x_blind[y_blind == 1])),
        "blind_negative_median": float(np.median(x_blind[y_blind == 0])),
    })
result = pd.DataFrame(rows).sort_values("blind_oriented_auc", ascending=False)
result.to_csv(output_path, index=False)
print(result.to_string(index=False))
