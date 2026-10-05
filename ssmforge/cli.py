"""SSMForge command-line interface.

Examples
--------
    python -m ssmforge.cli run --task adding --T 1000
    python -m ssmforge.cli benchmark --quick
    python -m ssmforge.cli ablation --task adding --T 1000
"""

from __future__ import annotations

import argparse
import json
import sys

from ssmforge.core.config import Config
from ssmforge.pipeline.pipeline import SSMPipeline, _fmt_row


def main(argv=None) -> int:
    p = argparse.ArgumentParser("ssmforge", description="SSMForge CLI")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-state", type=int, default=None)
    p.add_argument("--n-rates", type=int, default=None)
    p.add_argument("--lstm-hidden", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="train S4DFuse on one task")
    r.add_argument("--task", default="adding", choices=["adding", "longcopy", "seqclf"])
    r.add_argument("--T", type=int, default=1000)

    b = sub.add_parser("benchmark", help="full multi-seed benchmark")
    b.add_argument("--quick", action="store_true")

    a = sub.add_parser("ablation", help="multi-rate fusion ablation")
    a.add_argument("--task", default="adding")
    a.add_argument("--T", type=int, default=1000)

    args = p.parse_args(argv)
    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_state is not None:
        cfg.n_state = args.n_state
    if args.n_rates is not None:
        cfg.n_rates = args.n_rates
    if getattr(args, "lstm_hidden", None) is not None:
        cfg.lstm_hidden = args.lstm_hidden
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    pipe = SSMPipeline(cfg)

    if args.cmd == "run":
        print(json.dumps(pipe.run(args.task, args.T), indent=2))
    elif args.cmd == "benchmark":
        rep = pipe.benchmark(quick=args.quick)
        for row in rep["results"]:
            print(_fmt_row(row))
        print(f"\nelapsed {rep['elapsed_sec']}s")
    elif args.cmd == "ablation":
        print(json.dumps(pipe.ablation(args.task, args.T), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
