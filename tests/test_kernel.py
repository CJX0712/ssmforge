"""Core invariants of the S4D math kernel.

The most important test in the suite is ``test_conv_equals_recurrence``: the
convolution form and the recurrent form of the state-space model are algebraically
identical, so they must agree to machine precision. This is the cross-check that
validates the whole kernel implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssmforge.core.errors import SSMError
from ssmforge.ssm import kernel as K


def test_hippo_diagonal_properties():
    lam = K.hippo_legs_diagonal(16)
    assert lam.shape == (16,)
    assert np.iscomplexobj(lam)
    # strictly stable: every real part is negative
    assert np.all(lam.real < 0)
    # channel 0 is the slowest integrator: real part = -0.5
    assert lam[0].real == pytest.approx(-0.5)
    # imaginary parts are pi * n
    for n in (0, 3, 7, 15):
        assert lam[n].imag == pytest.approx(np.pi * n)


def test_kernel_shape_and_finiteness():
    lam = K.hippo_legs_diagonal(32)
    k = K.s4d_kernel(lam, 0.05, 100)
    assert k.shape == (100,)
    assert np.all(np.isfinite(k))


def test_kernel_rejects_nonpositive_delta():
    lam = K.hippo_legs_diagonal(8)
    with pytest.raises(SSMError):
        K.s4d_kernel(lam, 0.0, 10)


def test_conv_equals_recurrence():
    """Convolution form == recurrent form, to machine precision."""
    lam = K.hippo_legs_diagonal(32)
    rng = np.random.default_rng(0)
    u = rng.normal(size=256)
    for delta in (0.001, 0.05, 0.5):
        y_conv = K.conv1d_causal_real(u, K.s4d_kernel(lam, delta, 256))
        y_rec = K.recurrent_forward(lam, delta, u)
        assert np.max(np.abs(y_conv - y_rec)) < 1e-10


def test_conv_batched_matches_single():
    lam = K.hippo_legs_diagonal(16)
    rng = np.random.default_rng(1)
    U = rng.normal(size=(4, 128))
    k = K.s4d_kernel(lam, 0.05, 128)
    batched = K.conv1d_causal_real(U, k)
    for i in range(4):
        single = K.conv1d_causal_real(U[i], k)
        assert np.max(np.abs(batched[i] - single)) < 1e-10


def test_conv_is_causal():
    """Output at time t must not depend on future inputs."""
    lam = K.hippo_legs_diagonal(16)
    rng = np.random.default_rng(2)
    u = rng.normal(size=64)
    k = K.s4d_kernel(lam, 0.05, 64)
    y1 = K.conv1d_causal_real(u, k)
    u2 = u.copy()
    u2[40:] += 100.0  # perturb only the future
    y2 = K.conv1d_causal_real(u2, k)
    assert np.allclose(y1[:40], y2[:40], atol=1e-10)
    assert not np.allclose(y1[40:], y2[40:], atol=1e-6)


def test_fft_and_direct_paths_agree():
    """The long-sequence FFT path matches the direct path on a mid length."""
    lam = K.hippo_legs_diagonal(16)
    rng = np.random.default_rng(3)
    u = rng.normal(size=512)
    k = K.s4d_kernel(lam, 0.05, 512)
    direct = np.zeros(512)
    for s in range(512):
        direct[s] = u[s::-1] @ k.real[: s + 1]
    out = K.conv1d_causal_real(u, k)
    assert np.max(np.abs(out - direct)) < 1e-8


def test_integrator_channel_recovers_sum():
    """A near-integrator SSM channel, read at the last step, recovers sum(u).

    This is the mechanism S4DFuse uses to solve the adding problem: the slowest
    channel with a tiny rate is an approximate integrator, so its final state is
    (up to a constant scale) the total of the input.
    """
    lam = K.hippo_legs_diagonal(32)
    rng = np.random.default_rng(4)
    T = 400
    u = np.zeros((64, T))
    total = np.zeros(64)
    for i in range(64):
        v = rng.uniform(0, 1, 3)
        pos = rng.choice(T, 3, replace=False)
        u[i, pos] = v
        total[i] = v.sum()
    feats = K.recurrent_channels(lam, 1e-5, u, pool="last")
    integ = feats[:, 0]  # real part of channel 0 at the last step
    # least-squares slope for predicting `total` from `integ`:
    #   total ~ c * integ  =>  c = <integ, total> / <integ, integ>
    c = np.dot(integ, total) / np.dot(integ, integ)
    pred = c * integ
    # Residual is the small position-dependent decay of the near-integrator
    # (~0.2% over T=400), not a modelling error.
    assert np.max(np.abs(pred - total)) < 5e-3


def test_recurrent_channels_shapes():
    lam = K.hippo_legs_diagonal(8)
    rng = np.random.default_rng(5)
    u = rng.normal(size=(3, 50))
    for pool in ("mean", "last", "sum"):
        f = K.recurrent_channels(lam, 0.1, u, pool)
        assert f.shape == (3, 16)  # 2 * n_state (real + imag)
        assert np.all(np.isfinite(f))
