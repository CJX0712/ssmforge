"""Configuration with environment overrides and schema validation.

All tunables live here. ``ENV_SSMFORGE_*`` environment variables override the
constructor defaults so CI / Docker can parametrize without code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_SEED = 42


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@dataclass
class Config:
    """Global configuration with schema validation.

    Notes
    -----
    The forward pass is O(L) in sequence length for SSM models and O(L^2) for
    the attention baseline. ``max_len`` caps attention to keep it tractable.
    """

    seed: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_SEED", _DEFAULT_SEED))
    n_state: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_N_STATE", 32))
    n_rates: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_N_RATES", 4))
    pool: str = field(default_factory=lambda: os.environ.get("ENV_SSMFORGE_POOL", "last"))
    ridge_lambda: float = field(
        default_factory=lambda: _env_float("ENV_SSMFORGE_RIDGE_LAMBDA", 1e-3)
    )
    max_len: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_MAX_LEN", 256))
    lstm_hidden: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_LSTM_HIDDEN", 12))
    n_seeds: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_N_SEEDS", 3))
    base_seed: int = field(default_factory=lambda: _env_int("ENV_SSMFORGE_BASE_SEED", 1000))

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.n_state <= 0:
            raise ValueError("n_state must be > 0")
        if self.n_rates <= 0:
            raise ValueError("n_rates must be > 0")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if self.max_len <= 0:
            raise ValueError("max_len must be > 0")

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_state": self.n_state,
            "n_rates": self.n_rates,
            "pool": self.pool,
            "ridge_lambda": self.ridge_lambda,
            "max_len": self.max_len,
            "lstm_hidden": self.lstm_hidden,
            "n_seeds": self.n_seeds,
            "base_seed": self.base_seed,
        }
