"""SSMForge core package.

Common types, error codes, configuration, deterministic seeding, and the
public Protocol interfaces used across the system. The dependency graph is
strictly one-directional and acyclic:

    cli -> pipeline -> {data, training, ssm, eval} -> core
"""

from ssmforge.core.config import Config
from ssmforge.core.errors import ErrorCode, SSMError
from ssmforge.core.interfaces import BenchmarkResult, SequenceModel, SSMKernel
from ssmforge.core.seed import set_all

__all__ = [
    "BenchmarkResult",
    "Config",
    "ErrorCode",
    "SSMError",
    "SSMKernel",
    "SequenceModel",
    "set_all",
]
