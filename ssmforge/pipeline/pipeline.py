"""SSMPipeline — the single entry point wiring data -> models -> benchmark.

Also hosts the **ablation** harness: it toggles the flagship's multi-rate fusion
(single fixed rate vs. the full log-spaced multi-rate bank) to prove where the
gain comes from. Ablations deliberately reuse the *same* fitting code path as
the main benchmark so the two can never drift apart.
"""

from __future__ import annotations

import numpy as np

from ssmforge.core.config import Config
from ssmforge.core.seed import set_all
from ssmforge.data.synthetic import build_dataset
from ssmforge.eval.benchmark import run_benchmark
from ssmforge.eval.metrics import evaluate
from ssmforge.ssm import S4DFuse
from ssmforge.ssm.s4dfuse import _bank_rates


class SSMPipeline:
    """End-to-end pipeline for the SSMForge system."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def run(self, task: str = "adding", T: int = 1000, seed: int | None = None) -> dict:
        """Train S4DFuse on one task and return a single-run report."""
        cfg = self.config
        s = cfg.seed if seed is None else seed
        set_all(s)
        rng = np.random.default_rng(s)
        ds = build_dataset(task, rng, T=T)
        mdl = S4DFuse(
            n_state=cfg.n_state,
            n_rates=cfg.n_rates,
            pool=cfg.pool,
            ridge_lambda=cfg.ridge_lambda,
            seed=s,
        )
        mdl.fit(ds.X_train, ds.y_train)
        pred = mdl.predict(ds.X_test)
        metric, val = evaluate(ds.task, ds.y_test, pred)
        return {
            "task": f"{task}_T{T}",
            "model": "S4DFuse",
            "metric": metric,
            "value": val,
            "seed": s,
        }

    def benchmark(self, quick: bool = False) -> dict:
        """Full multi-seed benchmark sweep (writes nothing; returns dict)."""
        return run_benchmark(self.config, quick=quick)

    def ablation(self, task: str = "adding", T: int = 1000, n_seeds: int | None = None) -> dict:
        """Ablate the multi-rate fusion: full bank vs. single fixed rate.

        Arm A (full)    : full multi-rate bank  -- the flagship.
        Arm B (single)  : every rate collapsed to the geometric-mean rate
                          (degenerates to a single-rate S4D-Real layer).
        Arm C (one_rate): a single rate only.
        """
        cfg = self.config
        seeds = range(cfg.base_seed, cfg.base_seed + (n_seeds or cfg.n_seeds))
        arms = {
            "full_multirate": lambda r: _bank_rates(r),
            "single_rate": lambda r: np.full(r, np.sqrt(0.002 * 0.5)),
            "one_rate": lambda r: _bank_rates(1)[:1],
        }
        out = {}
        for arm, rate_fn in arms.items():
            vals = []
            for s in seeds:
                set_all(s)
                rng = np.random.default_rng(s)
                ds = build_dataset(task, rng, T=T)
                rates_n = 1 if arm == "one_rate" else cfg.n_rates
                mdl = S4DFuse(
                    n_state=cfg.n_state,
                    n_rates=rates_n,
                    pool=cfg.pool,
                    ridge_lambda=cfg.ridge_lambda,
                    seed=s,
                    rates=list(rate_fn(rates_n)),
                )
                mdl.fit(ds.X_train, ds.y_train)
                _, val = evaluate(ds.task, ds.y_test, mdl.predict(ds.X_test))
                vals.append(val)
            out[arm] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_seeds": len(vals),
            }
        return out


def _fmt_row(r: dict) -> str:
    if not r["available"]:
        return f"  {r['model']:<14} {r['task']:<12} {r['metric']:<9}     skipped  ({r['notes']})"
    return f"  {r['model']:<14} {r['task']:<12} {r['metric']:<9} {r['value']:.4f} ± {r['std']:.4f}"
