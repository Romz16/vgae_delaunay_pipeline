"""Metric helpers used by training and reporting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro-F1.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        Dictionary with ``acc`` and ``f1``.
    """
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def metric_summary(values: Iterable[float]) -> dict[str, float]:
    """Return mean, population standard deviation and 95% confidence interval.

    Args:
        values: Numeric values from repeated runs.

    Returns:
        Dictionary with ``mean``, ``std`` and ``ci95``.
    """
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
    """Format a metric like the previous experiment CSV files.

    Args:
        mean: Mean metric value in [0, 1].
        std: Standard deviation in [0, 1].
        ci95: 95% confidence interval half-width in [0, 1].

    Returns:
        Formatted percentage string.
    """
    return f"{mean * 100:.2f}    {std * 100:.2f} (CI95:   {ci95 * 100:.2f})"
