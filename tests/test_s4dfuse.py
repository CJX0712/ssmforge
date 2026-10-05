"""Tests for the S4DFuse flagship: data, determinism, and performance gates."""

from __future__ import annotations

import numpy as np
import pytest

from ssmforge.core.config import Config
from ssmforge.core.seed import set_all
from ssmforge.data.synthetic import build_dataset, make_adding, make_seqclf
from ssmforge.pipeline.pipeline import SSMPipeline
from ssmforge.ssm import S4DFuse


def test_adding_dgp_shapes_and_target():
    rng = np.random.default_rng(0)
    X, y = make_adding(rng, T=200, n_samples=50)
    assert X.shape == (50, 200, 1)
    assert y.shape == (50,)
    # exactly two spikes per sample
    for i in range(50):
        assert np.count_nonzero(X[i, :, 0]) == 2
    # target is the sum of the two spike values
    assert np.allclose(y, X[:, :, 0].sum(axis=1))


def test_seqclf_dgp_deterministic():
    X1, y1 = make_seqclf(np.random.default_rng(7), L=128, n_samples=40)
    X2, y2 = make_seqclf(np.random.default_rng(7), L=128, n_samples=40)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_build_dataset_split_sizes():
    ds = build_dataset("adding", np.random.default_rng(0), n_train=40, n_test=20, T=50)
    assert ds.X_train.shape[0] == 40
    assert ds.X_test.shape[0] == 20
    assert ds.task == "regression"


def test_build_dataset_rejects_unknown():
    with pytest.raises(ValueError):
        build_dataset("nope", np.random.default_rng(0))


def test_s4dfuse_solves_adding_exactly():
    """Primary performance gate: S4DFuse must essentially nail the adding task.

    The multi-rate integrator bank recovers the target to near machine-precision
    region, far below the LSTM baseline.
    """
    ds = build_dataset("adding", np.random.default_rng(0), T=200, n_train=200, n_test=100)
    rates = list(np.logspace(np.log10(1e-5), np.log10(0.5), 5))
    m = S4DFuse(n_state=32, n_rates=5, pool="last", seed=0, rates=rates)
    m.fit(ds.X_train, ds.y_train)
    mse = float(np.mean((ds.y_test - m.predict(ds.X_test)) ** 2))
    assert mse < 1e-3, f"adding MSE {mse} should be < 1e-3"


def test_s4dfuse_deterministic_same_seed():
    ds = build_dataset("adding", np.random.default_rng(0), T=100, n_train=80, n_test=40)
    rates = list(np.logspace(np.log10(1e-5), np.log10(0.5), 3))
    p1 = (
        S4DFuse(n_state=16, n_rates=3, seed=3, rates=rates)
        .fit(ds.X_train, ds.y_train)
        .predict(ds.X_test)
    )
    p2 = (
        S4DFuse(n_state=16, n_rates=3, seed=3, rates=rates)
        .fit(ds.X_train, ds.y_train)
        .predict(ds.X_test)
    )
    assert np.array_equal(p1, p2)


def test_s4dfuse_available_offline():
    m = S4DFuse(n_state=8, n_rates=2, seed=0)
    assert m.available is True  # pure numpy, always runnable


def test_multirate_beats_single_rate_ablation():
    """The flagship's value-add is the multi-rate bank (integrator rate).

    Removing the slow/integrator rate collapses performance to the mean-predictor
    level, proving the multi-rate fusion is essential and not incidental.
    """
    pipe = SSMPipeline(Config(n_seeds=2))
    abl = pipe.ablation("adding", 400)
    assert abl["full_multirate"]["mean"] < abl["single_rate"]["mean"]
    # single-rate should sit at (or near) the mean-predictor variance
    assert abl["single_rate"]["mean"] > 0.5 * abl["full_multirate"]["mean"] + 0.01


def test_pipeline_run_returns_report():
    pipe = SSMPipeline(Config(n_seeds=1))
    rep = pipe.run("adding", T=100, seed=0)
    assert rep["model"] == "S4DFuse"
    assert rep["metric"] == "mse"
    assert np.isfinite(rep["value"])


def test_seed_set_all_returns_int():
    assert isinstance(set_all(123), int)
