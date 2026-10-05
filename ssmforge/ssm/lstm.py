"""LSTM baseline implemented from scratch in NumPy (BPTT, end-to-end SGD).

This is the strong recurrent baseline. It is trained fully (last-hidden readout
learned by SGD) so the comparison against S4DFuse is fair. We recompute gate
pre-activations during the backward pass to keep memory linear in the sequence
length rather than storing all gate tensors.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.errors import ErrorCode, SSMError
from ssmforge.core.seed import set_all
from ssmforge.core.types import is_multiclass_target


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class LSTM:
    """Minimal but correct single-layer LSTM with BPTT."""

    name = "LSTM"

    def __init__(
        self,
        hidden: int = 16,
        lr: float = 0.02,
        epochs: int = 100,
        clip: float = 5.0,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.clip = clip
        self.seed = seed
        self.available = True
        self._is_classification = False

    def _init_params(self, F: int, out: int) -> None:
        rng = np.random.default_rng(self.seed)
        lim = 1.0 / np.sqrt(self.hidden)
        self.Wx = rng.uniform(-lim, lim, (4 * self.hidden, F))
        self.Wh = rng.uniform(-lim, lim, (4 * self.hidden, self.hidden))
        self.b = np.zeros(4 * self.hidden)
        # Forget-gate bias = 1 (Jozefowicz et al., 2015). Without this the cell
        # starts near-fully-forgeting and BPTT gradients die, so the LSTM
        # collapses to predicting the mean and never learns the task.
        self.b[self.hidden : 2 * self.hidden] = 1.0
        self.Why = rng.uniform(-lim, lim, (out, 2 * self.hidden))
        self.by = np.zeros(out)

    def _z(self, X, hprev):
        return X @ self.Wx.T + hprev @ self.Wh.T + self.b

    def _forward(self, X):
        B, L, F = X.shape
        H = self.hidden
        h_seq = np.zeros((B, L, H))
        c_seq = np.zeros((B, L, H))
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        for t in range(L):
            z = self._z(X[:, t, :], h)
            i = _sigmoid(z[:, 0:H])
            f = _sigmoid(z[:, H : 2 * H])
            g = np.tanh(z[:, 2 * H : 3 * H])
            o = _sigmoid(z[:, 3 * H : 4 * H])
            c = f * c + i * g
            h = o * np.tanh(c)
            h_seq[:, t, :] = h
            c_seq[:, t, :] = c
        return h_seq, c_seq

    def _loss_and_grad(self, X, y):
        B, L, F = X.shape
        H = self.hidden
        h_seq, c_seq = self._forward(X)
        h_last = h_seq[:, -1, :]
        c_last = c_seq[:, -1, :]
        # Read from BOTH the last hidden and last cell state. The cell state is
        # an unbounded linear register; the tanh-bounded hidden state alone
        # cannot represent a sum that exceeds 1 (the adding problem's target
        # reaches 2), so a hidden-only readout is structurally handicapped.
        R = np.concatenate([h_last, c_last], axis=1)  # (B, 2H)
        out = R @ self.Why.T + self.by  # (B, out)

        if self._is_classification:
            out = out - out.max(axis=1, keepdims=True)
            e = np.exp(out)
            p = e / e.sum(axis=1, keepdims=True)
            loss = -np.mean(np.log(p[np.arange(B), y.astype(int)] + 1e-12))
            dout = p.copy()
            dout[np.arange(B), y.astype(int)] -= 1.0
            dout /= B
        else:
            if y.ndim == 1:
                y = y[:, None]
            loss = 0.5 * np.mean((out - y) ** 2)
            dout = (out - y) / B

        dWhy = dout.T @ R
        dby = dout.sum(axis=0)
        dh = dout @ self.Why[:, :H]  # grad into h_last (hidden part of readout)
        # grad into c_last (cell part of readout) seeds the cell-gradient carry
        dc = dout @ self.Why[:, H:]

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        for t in reversed(range(L)):
            hprev = h_seq[:, t - 1, :] if t > 0 else np.zeros((B, H))
            cprev = c_seq[:, t - 1, :] if t > 0 else np.zeros((B, H))
            z = self._z(X[:, t, :], hprev)
            i = _sigmoid(z[:, 0:H])
            f = _sigmoid(z[:, H : 2 * H])
            g = np.tanh(z[:, 2 * H : 3 * H])
            o = _sigmoid(z[:, 3 * H : 4 * H])
            tanhc = np.tanh(c_seq[:, t, :])
            dc_t = dh * (o * (1.0 - tanhc**2)) + dc
            di = dc_t * g
            dg = dc_t * i
            df = dc_t * cprev
            dz_i = di * i * (1 - i)
            dz_f = df * f * (1 - f)
            dz_g = dg * (1 - g**2)
            dz_o = (dh * tanhc) * o * (1 - o)
            dz = np.concatenate([dz_i, dz_f, dz_g, dz_o], axis=1)
            dWx += dz.T @ X[:, t, :]
            if t > 0:
                dWh += dz.T @ hprev
            db += dz.sum(axis=0)
            dc = dc_t * f
            dh = dz @ self.Wh
        self._clip(dWx, dWh, db, dWhy, dby)
        return loss, dWx, dWh, db, dWhy, dby

    def _clip(self, *grads):
        for g in grads:
            np.clip(g, -self.clip, self.clip, out=g)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSTM:
        set_all(self.seed)
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, :, None]
        y = np.asarray(y)
        B, L, F = X.shape
        classes = np.unique(y)
        self._is_classification = is_multiclass_target(y)
        if self._is_classification:
            self._classes = classes
            out_dim = classes.size
            y_idx = np.searchsorted(classes, y)
            self._y_mean = 0.0
            self._y_std = 1.0
        else:
            out_dim = 1 if y.ndim == 1 else y.shape[1]
            y_idx = y.astype(np.float64)
            # Standardize targets so SGD is well-conditioned (a raw target with
            # mean ~1 makes the output layer collapse to a constant).
            self._y_mean = float(y_idx.mean())
            self._y_std = float(y_idx.std()) or 1.0
            y_idx = (y_idx - self._y_mean) / self._y_std
        self._init_params(F, out_dim)
        lr = self.lr
        for _ in range(self.epochs):
            loss, dWx, dWh, db, dWhy, dby = self._loss_and_grad(X, y_idx)
            self.Wx -= lr * dWx
            self.Wh -= lr * dWh
            self.b -= lr * db
            self.Why -= lr * dWhy
            self.by -= lr * dby
            lr *= 0.995
        if not np.all(np.isfinite(self.Wx)):
            raise SSMError(ErrorCode.DIVERGED, "LSTM diverged")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[:, :, None]
        h_seq, c_seq = self._forward(X)
        R = np.concatenate([h_seq[:, -1, :], c_seq[:, -1, :]], axis=1)
        out = R @ self.Why.T + self.by
        if self._is_classification:
            return self._classes[np.argmax(out, axis=1)]
        out = out * self._y_std + self._y_mean
        return out[:, 0] if out.shape[1] == 1 else out
