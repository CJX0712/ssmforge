"""Ridge readout head (dual form, train-only standardization).

The SSM produces linear features; the optimal linear readout is a ridge
regression. We use the **dual** (Woodbury) formulation so it works whether the
feature dimension P is larger or smaller than the sample count N, and we
standardize features using *train statistics only* (no leakage). Handles
regression, binary, and multiclass (one-vs-rest) targets uniformly.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.errors import ErrorCode, SSMError
from ssmforge.core.types import is_multiclass_target


class RidgeHead:
    """Deterministic ridge readout."""

    def __init__(self, lam: float = 1e-3):
        self.lam = float(lam)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._W: np.ndarray | None = None  # (P, K)
        self._mode: str = "direct"  # "direct" | "ovr"

    def _standardize(self, Z: np.ndarray, fit: bool) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        if fit:
            self._mu = Z.mean(axis=0)
            self._sigma = Z.std(axis=0)
            self._sigma[self._sigma < 1e-12] = 1.0
        return (Z - self._mu) / self._sigma

    def fit(self, Z: np.ndarray, y: np.ndarray) -> RidgeHead:
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y)
        N, P = Z.shape
        Zs = self._standardize(Z, fit=True)
        # Append a constant column so the readout can learn an intercept. Without
        # it, a zero-mean feature matrix can only produce zero-mean predictions
        # and the fit is silently offset by mean(y).
        Za = np.hstack([Zs, np.ones((N, 1))])
        Pa = P + 1

        # Determine output layout (robust multiclass detection — see
        # core.types.is_multiclass_target; a continuous regression target must
        # NOT be mistaken for an N-way classifier).
        if is_multiclass_target(y):
            self._mode = "ovr"
            classes = np.unique(y)
            K = classes.size
            Y = np.zeros((N, K), dtype=np.float64)
            for k in range(K):
                Y[:, k] = (y == classes[k]).astype(np.float64)
            self._classes = classes
        else:
            self._mode = "direct"
            Y = y.reshape(N, -1).astype(np.float64)

        # Dual ridge: solve via the cheaper of (Pa,Pa) or (N,N).
        if Pa <= N:
            A = Za.T @ Za + self.lam * np.eye(Pa)
            W = np.linalg.solve(A, Za.T @ Y)
        else:
            G = Za @ Za.T  # (N, N)
            A = G + self.lam * np.eye(N)
            alpha = np.linalg.solve(A, Y)  # (N, K)
            W = Za.T @ alpha  # (Pa, K)
        if not np.all(np.isfinite(W)):
            raise SSMError(ErrorCode.SINGULAR_MATRIX, "ridge produced non-finite W")
        self._W = W
        return self

    def _augment(self, Z: np.ndarray) -> np.ndarray:
        Zs = self._standardize(Z, fit=False)
        return np.hstack([Zs, np.ones((Zs.shape[0], 1))])

    def predict_scores(self, Z: np.ndarray) -> np.ndarray:
        if self._W is None:
            raise SSMError(ErrorCode.INVALID_CONFIG, "RidgeHead not fitted")
        return self._augment(Z) @ self._W  # (N, K)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(Z)
        if self._mode == "ovr":
            return self._classes[np.argmax(scores, axis=1)]
        if scores.shape[1] == 1:
            return scores[:, 0]
        return scores  # regression / direct multi-output
