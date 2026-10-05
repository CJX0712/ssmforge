"""Evaluation metrics.

Regression  -> MSE (lower is better).
Classification -> accuracy (higher is better).

The benchmark harness keeps the direction of "better" explicit so no comparison
in the report can silently flip sign.
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def evaluate(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Return ``(metric_name, value)`` for the given task."""
    if task == "regression":
        return "mse", mse(y_true, y_pred)
    return "accuracy", accuracy(y_true, y_pred)


def lower_is_better(metric: str) -> bool:
    return metric == "mse"


def beats(metric: str, flagship: float, baseline: float) -> bool:
    """True when ``flagship`` is strictly better than ``baseline``."""
    if lower_is_better(metric):
        return flagship < baseline
    return flagship > baseline
