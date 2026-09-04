"""CLI for golden-set runs, report comparison, and the AD-11 gate.

Examples:
    python -m scripts.run_eval --name loop-off --loop-mode off
    python -m scripts.run_eval --name loop-on --loop-mode on
    python -m scripts.run_eval --compare loop-off loop-on
    python -m scripts.run_eval --gate loop-off loop-on --shadow
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_harness import compare, run_fn_harness, run_harness  # noqa: E402
from evals.golden_set import GOLDEN_SET  # noqa: E402
from src.db import init_db  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts.run_eval")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--compare", nargs=2, metavar=("BASELINE", "CANDIDATE")
    )
    action.add_argument("--gate", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--shadow", action="store_true", help="advisory gate mode")
    parser.add_argument("--name", default="latest", help="saved report name")
    parser.add_argument(
        "--loop-mode",
        choices=("off", "on"),
        default="on",
        help="freeze bounded reflection off or on for this report",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.compare:
        compare(*args.compare)
        return
    if args.gate:
        from evals.gate import main as gate_main

        gate_argv = [*args.gate]
        if args.shadow:
            gate_argv.insert(0, "--shadow")
        raise SystemExit(gate_main(gate_argv))
    if args.shadow:
        _parser().error("--shadow requires --gate")

    init_db()
    run_harness(
        GOLDEN_SET,
        run_fn=partial(run_fn_harness, loop_mode=args.loop_mode),
        name=args.name,
    )


if __name__ == "__main__":
    main()
