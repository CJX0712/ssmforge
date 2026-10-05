"""SSMForge — Structured State Space Models (S4/S4D) research toolkit.

Flagship: :class:`~ssmforge.ssm.s4dfuse.S4DFuse`, a multi-rate S4D-Real model.
"""

__version__ = "0.1.0"
__author__ = "晨星 (CJX0712)"

from ssmforge.core import Config, set_all
from ssmforge.pipeline import SSMPipeline
from ssmforge.ssm import LSTM, ARRidge, AttentionRef, EMAMemory, S4DFuse

__all__ = [
    "LSTM",
    "ARRidge",
    "AttentionRef",
    "Config",
    "EMAMemory",
    "S4DFuse",
    "SSMPipeline",
    "__version__",
    "set_all",
]
