"""Error taxonomy for SSMForge.

Every failure carries a stable ``ErrorCode`` (E1xx..E5xx) so that the pipeline,
CLI and tests can branch deterministically. Importing this module must never
raise and must not import heavy dependencies.
"""

from __future__ import annotations

from enum import IntEnum


class ErrorCode(IntEnum):
    """Stable error codes (do not reuse / renumber)."""

    # E1xx: configuration & input validation
    INVALID_CONFIG = 100
    INVALID_SHAPE = 101
    UNKNOWN_DATASET = 102
    UNKNOWN_MODEL = 103

    # E2xx: numerical / math kernel
    KERNEL_NAN = 200
    DISCRETIZATION_FAILED = 201
    SINGULAR_MATRIX = 202

    # E3xx: training / optimization
    DIVERGED = 300
    CONVERGENCE_FAILED = 301

    # E4xx: evaluation / benchmark
    BENCHMARK_FAILED = 400
    INVARIANT_VIOLATED = 401

    # E5xx: environment / backend
    BACKEND_UNAVAILABLE = 500
    DETERMINISM_VIOLATED = 501


class SSMError(Exception):
    """Base exception carrying an :class:`ErrorCode`."""

    def __init__(self, code: ErrorCode, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"[{code.name}] {message}")

    @classmethod
    def from_code(cls, code: ErrorCode, message: str = "") -> SSMError:
        return cls(code, message)
