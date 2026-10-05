"""Tests for the ridge readout, target-type detection, and baselines."""

from __future__ import annotations

import numpy as np

from ssmforge.core.types import is_multiclass_target
from ssmforge.ssm import LSTM, ARRidge, EMAMemory
from ssmforge.training.ridge import RidgeHead


def test_multiclass_detection_rejects_continuous_regression():
    """A continuous target with many unique values must NOT be read as classes.

    Regression on the adding problem yields sums with as many unique values as
    samples; a naive ``unique(y).size > 2`` test would build an N-way classifier.
    """
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 2, size=300)  # continuous, 300 unique
    assert np.unique(y).size == 300
    assert is_multiclass_target(y) is False


def test_multiclass_detection_accepts_labels():
    y = np.repeat(np.arange(10), 30)  # 10 integer classes
    assert is_multiclass_target(y) is True


def test_multiclass_detection_binary_is_direct():
    y = np.array([0, 1] * 50)
    assert is_multiclass_target(y) is False  # binary stays in direct branch


def test_ridge_learns_intercept():
    """Features are zero-mean after standardization; the fit must recover a bias.

    Without an intercept column, a zero-mean design matrix can only produce
    zero-mean predictions and the fit is silently offset by mean(y).
    """
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(200, 3))
    y = 5.0 + 2.0 * Z[:, 0] + rng.normal(0, 0.1, 200)
    head = RidgeHead(lam=1e-6).fit(Z, y)
    pred = head.predict(Z)
    # The intercept must be recovered (prediction mean matches target mean) and
    # the fit must be tight up to the injected noise.
    assert abs(pred.mean() - y.mean()) < 0.05
    assert float(np.mean((pred - y) ** 2)) < 0.05


def test_ridge_dual_matches_primal():
    """The dual (Woodbury) solve must equal the primal solve.

    Both are computed on the *same* standardized features with the same bias
    column that RidgeHead uses internally.
    """
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(40, 60))  # P > N -> dual path
    w_true = rng.normal(size=60)
    y = Z @ w_true + 0.1 * rng.normal(size=40)
    lam = 1.0
    head = RidgeHead(lam=lam).fit(Z, y)
    pred_dual = head.predict(Z)
    # explicit primal on the same (standardized + bias) design matrix
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Zs = (Z - mu) / sigma
    Za = np.hstack([Zs, np.ones((Zs.shape[0], 1))])
    W = np.linalg.solve(Za.T @ Za + lam * np.eye(Za.shape[1]), Za.T @ y)
    pred_primal = Za @ W
    assert np.max(np.abs(pred_dual - pred_primal)) < 1e-6


def test_ridge_multiclass_roundtrip():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(150, 4))
    y = rng.integers(0, 3, 150)
    head = RidgeHead(lam=1e-2).fit(Z, y)
    assert set(np.unique(head.predict(Z))).issubset({0, 1, 2})


def test_lstm_gradient_check():
    """Numerical gradient check of the hand-written BPTT."""
    rng = np.random.default_rng(4)
    lstm = LSTM(hidden=4, lr=0.0, epochs=0, seed=0)
    X = rng.normal(size=(3, 6, 1))
    y = rng.normal(size=3)
    lstm._is_classification = False
    lstm._init_params(1, 1)
    _, dWx, dWh, db, dWhy, dby = lstm._loss_and_grad(X, y)

    def numgrad(param):
        eps = 1e-6
        g = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            i = it.multi_index
            old = param[i]
            param[i] = old + eps
            lp = lstm._loss_and_grad(X, y)[0]
            param[i] = old - eps
            lm = lstm._loss_and_grad(X, y)[0]
            param[i] = old
            g[i] = (lp - lm) / (2 * eps)
            it.iternext()
        return g

    for param, grad in (
        (lstm.Wx, dWx),
        (lstm.Wh, dWh),
        (lstm.b, db),
        (lstm.Why, dWhy),
        (lstm.by, dby),
    ):
        ng = numgrad(param)
        rel = np.max(np.abs(ng - grad)) / max(1e-12, np.max(np.abs(ng)))
        assert rel < 1e-5, f"gradient rel err {rel}"


def test_lstm_regression_learns_short_adding():
    """On a *short* adding problem the LSTM must beat the mean predictor.

    This guards against a silently broken baseline: if the LSTM cannot learn the
    trivial short-range case, its long-range failure would be meaningless.
    """
    from ssmforge.data.synthetic import build_dataset

    ds = build_dataset("adding", np.random.default_rng(0), T=10, n_train=200, n_test=150)
    lstm = LSTM(hidden=16, lr=0.05, epochs=200, seed=0).fit(ds.X_train, ds.y_train)
    pred = lstm.predict(ds.X_test)
    mse = float(np.mean((ds.y_test - pred) ** 2))
    assert mse < np.var(ds.y_test)  # strictly better than predicting the mean


def test_offline_fallback_available():
    """EMAMemory is the pure-NumPy offline fallback and must always be available."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 20, 1))
    y = rng.normal(size=30)
    m = EMAMemory(seed=0)
    assert m.available is True
    m.fit(X, y)
    assert m.predict(X).shape == (30,)


def test_ar_ridge_solves_short_adding_via_global_sum():
    """Linear control: on short adding, ARRidge recovers the target (sum of input)."""
    from ssmforge.data.synthetic import build_dataset

    ds = build_dataset("adding", np.random.default_rng(0), T=100, n_train=200, n_test=150)
    m = ARRidge(ridge_lambda=1e-2, seed=0).fit(ds.X_train, ds.y_train)
    pred = m.predict(ds.X_test)
    assert float(np.mean((ds.y_test - pred) ** 2)) < 0.01
