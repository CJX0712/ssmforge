"""Public Protocol interfaces.

Models and kernels depend only on these abstract contracts, never on concrete
implementations. This keeps the benchmark harness able to swap any
:class:`SequenceModel` (SSM / LSTM / attention / AR / EMA fallback) without
knowing its internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SSMKernel(Protocol):
    """A state-space convolution kernel generator."""

    def kernel(self, length: int) -> np.ndarray:
        """Return the length-``length`` convolution kernel (complex)."""
        ...

    def recurrent_step(self, x_prev: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Advance the recurrent state by one step."""
        ...


@dataclass
class BenchmarkResult:
    """One model's benchmark numbers on a single task."""

    model: str
    task: str
    metric: str
    value: float
    std: float = 0.0
    n_seeds: int = 1
    notes: str = ""
    available: bool = True

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "task": self.task,
            "metric": self.metric,
            "value": self.value,
            "std": self.std,
            "n_seeds": self.n_seeds,
            "notes": self.notes,
            "available": self.available,
        }


@runtime_checkable
class SequenceModel(Protocol):
    """Any sequence-to-label model used in the benchmark."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> SequenceModel:
        """Train on ``X`` (B, L) or (B, L, F) and targets ``y`` (B,) or (B, C)."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in the model's native output space."""
        ...

    @property
    def available(self) -> bool:
        """False when a required backend is missing (offline fallback path)."""
        ...
