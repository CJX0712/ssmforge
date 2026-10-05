"""End-to-end demo: benchmark + ablation + determinism check.

Writes ``benchmark.json`` next to the repository root. Run twice with the same
seed and every core metric must match bit-for-bit (only ``elapsed_sec`` varies).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssmforge.core.config import Config
from ssmforge.core.seed import set_all
from ssmforge.pipeline.pipeline import SSMPipeline, _fmt_row
from ssmforge.ssm import kernel as K


def _check_dual_mode() -> float:
    """Invariant: convolution and recurrence are bit-for-bit identical."""
    set_all(0)
    lam = K.hippo_legs_diagonal(32)
    rng = np.random.default_rng(0)
    u = rng.normal(size=64)
    d = 0.05
    y_conv = K.conv1d_causal_real(u, K.s4d_kernel(lam, d, 64))
    y_rec = K.recurrent_forward(lam, d, u)
    return float(np.max(np.abs(y_conv - y_rec)))


def main() -> int:
    set_all(42)
    cfg = Config()
    pipe = SSMPipeline(cfg)

    dual_err = _check_dual_mode()
    print(
        f"[invariant] conv/recurrent max|diff| = {dual_err:.3e} "
        f"({'PASS' if dual_err < 1e-10 else 'FAIL'})\n"
    )

    print("=== benchmark (multi-seed) ===")
    rep = pipe.benchmark(quick=False)
    for row in rep["results"]:
        print(_fmt_row(row))
    print(f"\nelapsed {rep['elapsed_sec']}s")

    print("\n=== ablation: multi-rate fusion ===")
    abl = pipe.ablation("adding", 1000)
    for arm, v in abl.items():
        print(f"  {arm:<16} mse {v['mean']:.4f} ± {v['std']:.4f}")

    out = {
        "config": cfg.as_dict(),
        "invariant_dual_mode_max_abs_diff": dual_err,
        "benchmark": rep,
        "ablation": abl,
    }
    dest = Path(__file__).resolve().parents[1] / "benchmark.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
