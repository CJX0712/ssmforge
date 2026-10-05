"""Synthetic, fully self-contained sequence datasets.

No network download is required: every dataset is generated from a fixed random
seed with a known data-generating process (DGP), so results are reproducible and
the "ground truth" is analytically defined. This keeps the system offline-safe
and avoids the HuggingFace/ModelScope download trap.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.types import Dataset


def make_adding(rng: np.random.Generator, T: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """The classic *Adding Problem* (Hochreiter & Schmidhuber, 1997).

    Single channel, zero almost everywhere, with exactly **two spikes**
    carrying independent Uniform[0,1] values placed uniformly in the first half
    of the sequence. The target is the sum of the two spike values. This is the
    canonical long-range credit-assignment task: a model must learn to retain
    two distant scalar values and add them. Crucially the target is a *linear*
    functional of the input (a global sum), so a linear state-space model with a
    long-memory (near-integrating) kernel solves it exactly while a recurrent
    network must *learn* that behaviour through BPTT.

    Returns
    -------
    X : (n_samples, T, 1)
    y : (n_samples,)
    """
    X = np.zeros((n_samples, T, 1))
    half = max(1, T // 2)
    pos = np.stack([rng.choice(half, size=2, replace=False) for _ in range(n_samples)])
    v = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    rows = np.arange(n_samples)
    X[rows, pos[:, 0], 0] = v[:, 0]
    X[rows, pos[:, 1], 0] = v[:, 1]
    y = v.sum(axis=1)
    return X, y


def make_longcopy(
    rng: np.random.Generator, T: int, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single-spike long-range retrieval.

    A single channel that is zero everywhere except one position chosen
    uniformly over the **entire** sequence, which carries a Uniform[0,1] value.
    The target is that single value. The model must retrieve one distant scalar;
    the further away the spike sits, the harder the credit assignment.

    Returns
    -------
    X : (n_samples, T, 1)
    y : (n_samples,)
    """
    X = np.zeros((n_samples, T, 1))
    pos = rng.integers(0, T, size=n_samples)
    v = rng.uniform(0.0, 1.0, size=n_samples)
    rows = np.arange(n_samples)
    X[rows, pos, 0] = v
    return X, v


def make_seqclf(
    rng: np.random.Generator,
    L: int = 784,
    n_samples: int = 1200,
    n_classes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic sMNIST-style sequential classification (no download).

    A 28x28-rasterised analogue: each sample is a length-``L`` scan of a
    class-specific low-frequency signal plus noise, with a class-dependent DC
    offset. The label is the class; recovering it requires **integrating** over
    the whole sequence (a global statistic), which is exactly what a state-space
    model with HiPPO structure does well and which a last-timestep baseline
    cannot do. This is a self-contained proxy that mirrors sMNIST's
    long-range-integration nature without fetching MNIST.

    Returns
    -------
    X : (n_samples, L)
    y : (n_samples,) int in [0, n_classes)
    """
    t = np.linspace(0.0, 1.0, L)
    X = np.zeros((n_samples, L))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        c = int(rng.integers(0, n_classes))
        freq = 1 + (c % 5)
        phase = (c / n_classes) * 2.0 * np.pi
        dc = 0.5 * (2.0 * (c / (n_classes - 1)) - 1.0)
        base = dc + 0.5 * np.sin(2.0 * np.pi * freq * t + phase)
        X[i] = base + rng.normal(0.0, 3.0, size=L)
        y[i] = c
    return X, y


def build_dataset(
    name: str,
    rng: np.random.Generator,
    n_train: int = 250,
    n_test: int = 150,
    T: int = 1000,
    L: int = 784,
) -> Dataset:
    """Build a named dataset split into train / test.

    Supported ``name``: ``adding``, ``longcopy``, ``seqclf``.
    """
    if name == "adding":
        Xtr, ytr = make_adding(rng, T, n_train)
        Xte, yte = make_adding(rng, T, n_test)
        return Dataset(name, Xtr, ytr, Xte, yte, "regression", {"T": T})
    if name == "longcopy":
        Xtr, ytr = make_longcopy(rng, T, n_train)
        Xte, yte = make_longcopy(rng, T, n_test)
        return Dataset(name, Xtr, ytr, Xte, yte, "regression", {"T": T})
    if name == "seqclf":
        Xtr, ytr = make_seqclf(rng, L=L, n_samples=n_train + n_test)
        Xte, yte = Xtr[n_train:], ytr[n_train:]
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
        return Dataset(name, Xtr, ytr, Xte, yte, "classification", {"L": L})
    raise ValueError(f"unknown dataset: {name}")
