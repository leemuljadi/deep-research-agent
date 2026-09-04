"""CLI: run the evaluation harness over the golden set.

Usage:
    python -m scripts.run_eval [--name <report-name>]
    python -m scripts.run_eval --compare <baseline> <candidate>
    python -m scripts.run_eval --gate <baseline> <candidate> [--shadow]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_harness import compare, run_fn_harness, run_harness  # noqa: E402
from evals.golden_set import GOLDEN_SET  # noqa: E402
from src.db import init_db  # noqa: E402


def main() -> None:
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        if len(sys.argv) < i + 3:
            print("Usage: python -m scripts.run_eval --compare <baseline> <candidate>")
            sys.exit(1)
        compare(sys.argv[i + 1], sys.argv[i + 2])
        return

    if "--gate" in sys.argv:
        i = sys.argv.index("--gate")
        if len(sys.argv) < i + 3:
            print("Usage: python -m scripts.run_eval --gate <baseline> <candidate> [--shadow]")
            sys.exit(1)
        # The merge gate (AD-11): same exit-code contract as `python -m
        # evals.gate` — 0 pass, 1 blocked, 3 EVAL_INFRA_FAILURE.
        from evals.gate import main as gate_main

        sys.exit(gate_main(sys.argv[i + 1 :]))  # remaining argv: candidate [--shadow]

    init_db()
    name = "latest"
    if "--name" in sys.argv:
        i = sys.argv.index("--name")
        name = sys.argv[i + 1]
    run_harness(GOLDEN_SET, run_fn=run_fn_harness, name=name)

if __name__ == "__main__":
    main()