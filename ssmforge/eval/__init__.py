"""Evaluation metrics and the multi-seed benchmark runner."""

from ssmforge.eval.benchmark import run_benchmark, run_task
from ssmforge.eval.metrics import accuracy, beats, evaluate, lower_is_better, mse

__all__ = [
    "accuracy",
    "beats",
    "evaluate",
    "lower_is_better",
    "mse",
    "run_benchmark",
    "run_task",
]
