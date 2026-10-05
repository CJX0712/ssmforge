"""Non-recurrent baselines and the pure-NumPy offline fallback.

* :class:`ARRidge`  — flattens the raw sequence and ridge-regresses. This is the
  "no temporal model" reference: it sees the whole input but ignores order, so
  it cannot solve the long-range tasks.
* :class:`EMAMemory` — the **offline fallback (Tier-1)**. A single exponential
  moving average per feature. It uses nothing but NumPy, so SSMForge always has
  a runnable degradation path even if SciPy / SciPy-backed extras are missing.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.seed import set_all
from ssmforge.training.ridge import RidgeHead


def _flat(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    return X.reshape(X.shape[0], -1)


class ARRidge:
    """Ridge regression on the raw flattened sequence (no temporal model)."""

    name = "ARRidge"

    def __init__(self, ridge_lambda: float = 1e-2, seed: int = 42):
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self._head = RidgeHead(lam=ridge_lambda)
        self.available = True

    def fit(self, X, y):
        set_all(self.seed)
        self._head.fit(_flat(X), y)
        return self

    def predict(self, X):
        return self._head.predict(_flat(X))


class EMAMemory:
    """Pure-NumPy exponential-moving-average memory + ridge readout.

    Offline Tier-1 fallback. h_t = a·h_{t-1} + (1-a)·u_t per feature; the
    flattened final state is passed to a ridge head.
    """

    name = "EMAMemory"

    def __init__(self, alpha: float = 0.9, ridge_lambda: float = 1e-2, seed: int = 42):
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self._head = RidgeHead(lam=ridge_lambda)
        self.available = True  # always available: pure NumPy

    def _features(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, :, None]
        B, L, F = X.shape
        h = np.zeros((B, F))
        for t in range(L):
            h = self.alpha * h + (1.0 - self.alpha) * X[:, t, :]
        return h

    def fit(self, X, y):
        set_all(self.seed)
        self._head.fit(self._features(X), y)
        return self

    def predict(self, X):
        return self._head.predict(self._features(X))
