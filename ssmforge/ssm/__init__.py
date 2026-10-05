"""SSM sequence models: the S4DFuse flagship plus baselines."""

from ssmforge.ssm.baselines import ARRidge, EMAMemory
from ssmforge.ssm.kernel import (
    conv1d_causal_real,
    hippo_legs_diagonal,
    recurrent_forward,
    s4d_kernel,
)
from ssmforge.ssm.lstm import LSTM
from ssmforge.ssm.s4dfuse import S4DFuse
from ssmforge.ssm.transformer import AttentionRef

__all__ = [
    "LSTM",
    "ARRidge",
    "AttentionRef",
    "EMAMemory",
    "S4DFuse",
    "conv1d_causal_real",
    "hippo_legs_diagonal",
    "recurrent_forward",
    "s4d_kernel",
]
