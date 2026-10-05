"""S4D-Real diagonal state-space kernel.

Continuous SSM:  x'(t) = A x(t) + B u(t),  y(t) = C x(t)
with A a *complex* diagonal (HiPPO-LegS), B = 1 (real), C = 1 (absorbed into
the downstream ridge readout), so each "head" differs only by its
discretization rate Δ. Zero-order-hold discretization gives a closed-form
causal convolution kernel:

    k_t = Σ_n  Ā_n^t · B̄_n,   Ā_n = exp(Δ·Λ_n),  B̄_n = (Ā_n - 1) / Λ_n

Because A is diagonal this is exact and element-wise. The real part of the
complex convolution output is a valid real LTI system (the standard S4D-Real
"take-real-part" realization). The recurrent form

    x_t = Ā x_{t-1} + B̄ u_t ,  y_t = Re(C · x_t)

is algebraically identical to the convolution and is verified bit-for-bit in
the test-suite.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.errors import ErrorCode, SSMError


def hippo_legs_diagonal(n_state: int) -> np.ndarray:
    """HiPPO-LegS diagonal Λ of length ``n_state`` (complex).

    Λ_n = -0.5·(2n + 1) + i·π·n     (the S4D-LegS / "complex" S4D init).
    """
    n = np.arange(n_state)
    real = -0.5 * (2.0 * n + 1.0)
    imag = np.pi * n
    return (real + 1j * imag).astype(np.complex128)


def s4d_kernel(lam: np.ndarray, delta: float, length: int) -> np.ndarray:
    """Closed-form S4D causal convolution kernel of length ``length``.

    Parameters
    ----------
    lam : (N,) complex
        Diagonal of A (HiPPO-LegS).
    delta : float
        Discretization step (> 0).
    length : int
        Kernel length L.

    Returns
    -------
    (L,) complex kernel ``k`` with ``k[t] = Σ_n Ā_n^t · (Ā_n - 1)/Λ_n``.
    """
    if delta <= 0:
        raise SSMError(ErrorCode.INVALID_CONFIG, "delta must be > 0")
    lam = np.asarray(lam, dtype=np.complex128)
    A_bar = np.exp(delta * lam)  # (N,) complex, |·|<1
    B_bar = (A_bar - 1.0) / lam  # (N,) complex
    t = np.arange(length, dtype=np.float64)  # (L,)
    # powers[n, t] = A_bar[n] ** t
    powers = A_bar[:, None] ** t[None, :]  # (N, L)
    kernel = (powers * B_bar[:, None]).sum(axis=0)  # (L,) complex
    if not np.all(np.isfinite(kernel)):
        raise SSMError(ErrorCode.KERNEL_NAN, "non-finite kernel")
    return kernel


def conv1d_causal_real(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Causal convolution of real signal ``u`` with (complex) kernel ``k``.

    Returns the **real part** of the complex output (S4D-Real realization).
    ``u`` is (L,) or (B, L); returns same-rank real array.

    Uses FFT for long sequences (fast, batched) and a direct time-domain sum for
    short ones (better numerics, no wraparound). Both paths are strictly causal
    (only ``y[0..L-1]`` is produced, using no future information).
    """
    u = np.asarray(u, dtype=np.float64)
    k = np.asarray(k, dtype=np.complex128)
    is_batch = u.ndim == 2
    U = u if is_batch else u[None, :]
    L = U.shape[-1]
    # Real input => only the real part of the complex kernel contributes:
    #   Re( sum_s U[s] * k[t-s] ) = sum_s U[s] * Re(k[t-s]).
    k_r = k.real

    # Direct path (accurate, small memory) for short sequences.
    if L <= 512:
        out = np.zeros((U.shape[0], L), dtype=np.float64)
        for s in range(L):
            out[:, s] = U[:, s::-1] @ k_r[: s + 1]
        y = out
    else:
        # FFT path (batched, O(N log N)) for long sequences.
        n_fft = 1 << int(np.ceil(np.log2(L + len(k_r) - 1)))
        Uf = np.fft.rfft(U, n=n_fft, axis=-1)
        Kf = np.fft.rfft(k_r, n=n_fft)
        full = np.fft.irfft(Uf * Kf[None, :], n=n_fft, axis=-1)
        y = full[:, :L]
    return y if is_batch else y[0]


def recurrent_channels(
    lam: np.ndarray, delta: float, u: np.ndarray, pool: str = "mean"
) -> np.ndarray:
    """Per-channel S4D responses, pooled over time (the S4DFuse feature bank).

    Instead of collapsing the N state channels into one scalar kernel (which
    buries the slowly-decaying integrating channel), we keep every channel
    separate. For each channel n the SSM state is

        x_n(t) = Ā_n x_n(t-1) + B̄_n u_t,     Ā_n = exp(Δ Λ_n),  B̄_n = (Ā_n-1)/Λ_n

    and the real feature is the concatenation of the real and imaginary parts
    of the (complex) state, pooled over t. The slowest channel (n = 0, whose
    real part of Λ is -0.5) acts as a near-integrator, so a mean-pooled linear
    readout can recover a running sum — exactly how S4 solves the adding problem.

    Parameters
    ----------
    lam : (N,) complex
    delta : float
    u : (B, L) real
    pool : "mean" | "last" | "sum"

    Returns
    -------
    (B, 2N) real features.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 1:
        u = u[None, :]
    lam = np.asarray(lam, dtype=np.complex128)
    A_bar = np.exp(delta * lam)  # (N,)
    B_bar = (A_bar - 1.0) / lam  # (N,)
    B, L = u.shape
    N = lam.shape[0]
    x = np.zeros((B, N), dtype=np.complex128)
    acc = np.zeros((B, N), dtype=np.complex128)
    for t in range(L):
        x = A_bar[None, :] * x + B_bar[None, :] * u[:, t][:, None]
        acc += x
    if pool == "mean":
        feat = acc / L
    elif pool == "sum":
        feat = acc
    elif pool == "last":
        feat = x
    else:
        raise ValueError(f"unknown pool: {pool}")
    return np.concatenate([feat.real, feat.imag], axis=1)  # (B, 2N)


def recurrent_forward(lam: np.ndarray, delta: float, u: np.ndarray) -> np.ndarray:
    """Recurrent SSM forward, returns real output (L,) or (B, L).

    x_t = Ā x_{t-1} + B̄ u_t,  y_t = Re(C · x_t)  with C = 1.
    """
    u = np.asarray(u, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.complex128)
    A_bar = np.exp(delta * lam)
    B_bar = (A_bar - 1.0) / lam
    N = lam.shape[0]
    if u.ndim == 1:
        x = np.zeros(N, dtype=np.complex128)
        out = np.zeros(u.shape[0], dtype=np.float64)
        for t in range(u.shape[0]):
            x = A_bar * x + B_bar * u[t]
            out[t] = x.real.sum()
        return out
    B, L = u.shape
    x = np.zeros((B, N), dtype=np.complex128)
    out = np.zeros((B, L), dtype=np.float64)
    for t in range(L):
        x = A_bar * x + B_bar * u[:, t][:, None]
        out[:, t] = x.real.sum(axis=1)
    return out
