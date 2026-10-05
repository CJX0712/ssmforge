"""Shared lightweight type aliases used across modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """A synthetic sequence dataset split into train / test."""

    name: str
    X_train: np.ndarray  # (Ntr, L) or (Ntr, L, F)
    y_train: np.ndarray  # (Ntr,) or (Ntr, C)
    X_test: np.ndarray
    y_test: np.ndarray
    task: str  # "regression" | "classification"
    meta: dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}


def as_2d(X: np.ndarray) -> np.ndarray:
    """Ensure ``X`` is (B, L). Append a feature dim if (B, L, 1)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim == 3 and X.shape[2] == 1:
        X = X[:, :, 0]
    return X


def is_multiclass_target(y: np.ndarray) -> bool:
    """Robustly decide whether ``y`` is a *class label* vector (multiclass).

    A naive ``np.unique(y).size > 2`` test is WRONG: a continuous regression
    target (e.g. the adding problem, whose sums are all distinct floats) has as
    many unique values as samples and would be misread as an N-way classifier.
    We therefore require the target to be integer-valued, to have a small
    number of distinct classes, and for those classes to be a minority of the
    samples. Binary {0,1} stays in the direct (regression) branch.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return True  # already one-hot / multi-output
    yf = y.astype(np.float64).reshape(-1)
    classes = np.unique(y)
    integer_valued = bool(np.all(np.equal(np.mod(yf, 1.0), 0.0)))
    small = classes.size <= 50
    minority = classes.size <= 0.5 * max(1, yf.size)
    return bool(integer_valued and classes.size > 2 and small and minority)
