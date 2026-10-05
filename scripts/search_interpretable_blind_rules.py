from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score

base_path = Path("outputs_revision/downloaded_results/corruption_recovery_pilot_20260918/graph_level_outcomes.csv")
blind_path = Path("outputs_revision/downloaded_results/blind_sequential_20260919/blind_all_frozen_predictions_scored.csv")
output_path = blind_path.parent / "blind_interpretable_rule_search.csv"

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


@dataclass
class Rule:
    text: str
    train_pred: np.ndarray
    blind_pred: np.ndarray
    used: tuple[str, ...]
    complexity: int

    @property
    def train_bacc(self):
        return balanced_accuracy_score(y_train, self.train_pred)


def valid_support(pred):
    fraction = np.mean(pred)
    return 0.08 <= fraction <= 0.80


atoms = []
for feature in features:
    train_x = pd.to_numeric(train[feature], errors="coerce")
    median = float(train_x.median())
    train_x = train_x.fillna(median).to_numpy(float)
    blind_x = pd.to_numeric(blind[feature], errors="coerce").fillna(median).to_numpy(float)
    for quantile in (0.20, 0.35, 0.50, 0.65, 0.80):
        threshold = float(np.quantile(train_x, quantile))
        for symbol, train_pred, blind_pred in (
            ("<=", train_x <= threshold, blind_x <= threshold),
            (">", train_x > threshold, blind_x > threshold),
        ):
            if valid_support(train_pred):
                atoms.append(Rule(f"{feature} {symbol} {threshold:.6g}", train_pred, blind_pred, (feature,), 1))

atoms.sort(key=lambda rule: (rule.train_bacc, -rule.complexity), reverse=True)
top_atoms = atoms[:35]

pairs = []
for i, left in enumerate(top_atoms):
    for right in top_atoms[i + 1:]:
        if left.used[0] == right.used[0]:
            continue
        for op in ("AND", "OR"):
            if op == "AND":
                train_pred = left.train_pred & right.train_pred
                blind_pred = left.blind_pred & right.blind_pred
            else:
                train_pred = left.train_pred | right.train_pred
                blind_pred = left.blind_pred | right.blind_pred
            if valid_support(train_pred):
                pairs.append(Rule(
                    f"({left.text}) {op} ({right.text})", train_pred, blind_pred,
                    tuple(sorted(set(left.used + right.used))), 2,
                ))
pairs.sort(key=lambda rule: (rule.train_bacc, -rule.complexity), reverse=True)
top_pairs = pairs[:60]

triples = []
for pair in top_pairs:
    for atom in top_atoms[:25]:
        if atom.used[0] in pair.used:
            continue
        for op in ("AND", "OR"):
            if op == "AND":
                train_pred = pair.train_pred & atom.train_pred
                blind_pred = pair.blind_pred & atom.blind_pred
            else:
                train_pred = pair.train_pred | atom.train_pred
                blind_pred = pair.blind_pred | atom.blind_pred
            if valid_support(train_pred):
                triples.append(Rule(
                    f"({pair.text}) {op} ({atom.text})", train_pred, blind_pred,
                    tuple(sorted(set(pair.used + atom.used))), 3,
                ))

all_rules = atoms + pairs + triples
# Selection uses old graphs only. A small complexity penalty avoids choosing a
# longer rule for a negligible training improvement.
selected = max(all_rules, key=lambda rule: (rule.train_bacc - 0.01 * (rule.complexity - 1), -rule.complexity))

def summarize(rule: Rule, selection_status: str):
    tn, fp, fn, tp = confusion_matrix(y_blind, rule.blind_pred, labels=[0, 1]).ravel()
    return {
        "selection_status": selection_status, "rule": rule.text, "complexity": rule.complexity,
        "train_balanced_accuracy": rule.train_bacc,
        "blind_balanced_accuracy": balanced_accuracy_score(y_blind, rule.blind_pred),
        "blind_accuracy": np.mean(rule.blind_pred == y_blind),
        "blind_precision": precision_score(y_blind, rule.blind_pred, zero_division=0),
        "blind_recall": recall_score(y_blind, rule.blind_pred, zero_division=0),
        "blind_specificity": tn / max(tn + fp, 1),
        "blind_predicted_positive_fraction": np.mean(rule.blind_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

rows = [summarize(selected, "selected_on_old_graphs")]
for rule in sorted(all_rules, key=lambda candidate: candidate.train_bacc, reverse=True)[:20]:
    rows.append(summarize(rule, "top20_on_old_graphs"))
result = pd.DataFrame(rows)
result.to_csv(output_path, index=False)
print(result.head(10).to_string(index=False))
