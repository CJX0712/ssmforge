"""Single-head self-attention baseline (NumPy, O(L^2)).

Reported as a *reference* competitor. For very long sequences the L x L
attention matrix is intractable on CPU, so the benchmark runner skips it
(``available=False``) rather than fabricating a number.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.seed import set_all
from ssmforge.training.ridge import RidgeHead


class AttentionRef:
    """Mean-pooled single-head self-attention + ridge readout."""

    name = "AttentionRef"

    def __init__(
        self, d_k: int = 32, ridge_lambda: float = 1e-3, seed: int = 42, max_len: int = 1000
    ):
        self.d_k = d_k
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.max_len = max_len
        self._head = RidgeHead(lam=ridge_lambda)
        self._Wq = None
        self._Wk = None
        self._Wv = None
        self._bias = None

    def _proj(self, X):
        rng = np.random.default_rng(self.seed)
        s = 1.0 / np.sqrt(X.shape[-1])
        self._Wq = rng.uniform(-s, s, (X.shape[-1], self.d_k))
        self._Wk = rng.uniform(-s, s, (X.shape[-1], self.d_k))
        self._Wv = rng.uniform(-s, s, (X.shape[-1], self.d_k))
        self._bias = rng.uniform(-s, s, (1, self.d_k))

    def _features(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, :, None]
        B, L, F = X.shape
        if self.max_len < L:
            raise MemoryError(f"attention L={L} > max_len={self.max_len}")
        Q = X @ self._Wq + self._bias
        K = X @ self._Wk
        V = X @ self._Wv
        scale = 1.0 / np.sqrt(self.d_k)
        scores = np.einsum("bld,bmd->blm", Q, K) * scale  # (B,L,L)
        scores -= scores.max(axis=-1, keepdims=True)
        A = np.exp(scores)
        A /= A.sum(axis=-1, keepdims=True)
        ctx = np.einsum("blm,bmd->bld", A, V)  # (B,L,d_k)
        return ctx.reshape(B, -1)  # mean-pool-free full context (B, L*d_k)

    def fit(self, X, y):
        set_all(self.seed)
        self._proj(np.asarray(X, dtype=np.float64))
        Z = self._features(X)
        self._head.fit(Z, y)
        return self

    def predict(self, X):
        return self._head.predict(self._features(X))
