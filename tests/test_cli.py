"""CLI smoke tests and config validation."""

from __future__ import annotations

import pytest

from ssmforge.cli import main
from ssmforge.core.config import Config
from ssmforge.core.errors import ErrorCode, SSMError


def test_config_defaults_valid():
    cfg = Config()
    d = cfg.as_dict()
    assert d["n_state"] > 0
    assert d["n_rates"] > 0
    assert d["max_len"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_state": 0},
        {"n_rates": 0},
        {"max_len": 0},
        {"ridge_lambda": -1.0},
        {"seed": -5},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_cli_run_smoke(capsys):
    rc = main(["--seed", "0", "run", "--task", "adding", "--T", "100"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "S4DFuse" in out


def test_cli_ablation_smoke(capsys):
    rc = main(
        [
            "--n-seeds",
            "1",
            "--n-rates",
            "2",
            "--n-state",
            "8",
            "ablation",
            "--task",
            "adding",
            "--T",
            "100",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "full_multirate" in out


def test_cli_benchmark_quick_smoke(capsys):
    rc = main(
        [
            "--n-seeds",
            "1",
            "--n-rates",
            "2",
            "--n-state",
            "8",
            "--lstm-hidden",
            "4",
            "benchmark",
            "--quick",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "S4DFuse" in out


def test_error_code_enum_stable():
    assert ErrorCode.INVALID_CONFIG == 100
    assert ErrorCode.BACKEND_UNAVAILABLE == 500


def test_ssm_error_message():
    e = SSMError(ErrorCode.KERNEL_NAN, "bad")
    assert "KERNEL_NAN" in str(e)
    assert e.code == ErrorCode.KERNEL_NAN
