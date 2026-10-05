"""S4DFuse — the SSMForge flagship.

A *multi-rate* diagonal state-space model. The construction is a bank of
independent S4D-Real state-space channels. The N state channels share the
HiPPO-LegS diagonal Λ (channel n has real decay -0.5(2n+1) and oscillation
πn), and the bank is replicated across several discretization rates Δ, so the
feature map spans a wide spectrum of memory horizons. Per-channel real and
imaginary state parts are pooled over time and combined by a ridge readout.

Because the SSM layer is fixed (HiPPO initialisation, B = 1) and only the linear
readout is trained, the model is fast, fully deterministic and never diverges —
yet the multi-rate, multi-channel bank gives the readout enough basis functions
to solve long-range tasks that recurrent networks must learn through BPTT.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.seed import set_all
from ssmforge.ssm.kernel import hippo_legs_diagonal, recurrent_channels
from ssmforge.training.ridge import RidgeHead


def _bank_rates(n_rates: int, r_min: float = 0.002, r_max: float = 0.5) -> np.ndarray:
    if n_rates == 1:
        return np.array([np.sqrt(r_min * r_max)])
    return np.logspace(np.log10(r_min), np.log10(r_max), n_rates)


class S4DFuse:
    """Multi-rate, multi-channel S4D-Real sequence model + ridge readout."""

    name = "S4DFuse"

    def __init__(
        self,
        n_state: int = 32,
        n_rates: int = 4,
        pool: str = "mean",
        ridge_lambda: float = 1e-3,
        seed: int = 42,
        rates: list[float] | None = None,
    ):
        self.n_state = n_state
        self.n_rates = n_rates
        self.pool = pool
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        if rates is not None and len(rates) == n_rates:
            self.deltas = np.asarray(rates, dtype=np.float64)
        else:
            self.deltas = _bank_rates(n_rates)
        self._lam = hippo_legs_diagonal(n_state)
        self._head = RidgeHead(lam=ridge_lambda)
        self._train_L: int | None = None
        self.available = True

    # ---- feature extraction -------------------------------------------------
    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        parts = []
        for d in self.deltas:
            if X.ndim == 3:
                # multi-channel input: sum per-input-channel SSM responses
                # (fixed linear input projection) at this rate.
                Bsz, L, F = X.shape
                z = np.zeros((Bsz, 2 * self.n_state))
                for f in range(F):
                    z += recurrent_channels(self._lam, d, X[:, :, f], self.pool)
            else:
                z = recurrent_channels(self._lam, d, X, self.pool)
            parts.append(z)
        # CONCATENATE across rates (never sum: summing would average the slow
        # integrating channel with the fast, noisy channels and destroy the
        # long-range signal). Feature dim = n_rates * 2 * n_state.
        return np.concatenate(parts, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> S4DFuse:
        set_all(self.seed)
        self._train_L = X.shape[1] if X.ndim >= 2 else X.shape[0]
        self._head.fit(self._features(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._head.predict(self._features(X))

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self._head.predict_scores(self._features(X))

    def n_features(self) -> int:
        return 2 * self.n_state
