"""Deterministic global seeding.

A single entry point ``set_all(seed)`` pins every randomness source used by the
system (Python ``random``, ``numpy`` global RNG, and any library-level global
RNG). The demo must produce bit-for-bit identical ``benchmark.json`` metrics
across two runs with the same seed (excluding wall-clock ``elapsed_sec``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

_DEFAULT_SEED = 42


def set_all(seed: int = _DEFAULT_SEED) -> int:
    """Pin all RNG sources to ``seed`` and return the active seed.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.

    Returns
    -------
    int
        The seed actually applied (clamped to ``[0, 2**32 - 1]``).
    """
    if seed is None:
        seed = _DEFAULT_SEED
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    try:  # library-level global RNG (best-effort; absent in pure-numpy builds)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


@dataclass(frozen=True)
class SeedState:
    """Captured seed state for audit / reproducibility reporting."""

    seed: int
    numpy_state: bytes

    @staticmethod
    def capture(seed: int) -> SeedState:
        return SeedState(seed=seed, numpy_state=np.random.get_state()[1].tobytes())
