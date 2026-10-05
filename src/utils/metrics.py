
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metric_summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    ci95 = float((std / np.sqrt(arr.size)) * 1.96)
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "ci95_lower": mean - ci95,
        "ci95_upper": mean + ci95,
    }


def format_metric(mean: float, std: float, ci95: float) -> str:
    return f"{mean * 100:.2f}    {std * 100:.2f} (CI95:   {ci95 * 100:.2f})"
