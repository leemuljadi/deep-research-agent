"""CLI: run the evaluation harness over the golden set.

Usage:
    python -m scripts.run_eval [--name <report-name>]
    python -m scripts.run_eval --compare <baseline> <candidate>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_harness import compare, run_harness  # noqa: E402
from evals.golden_set import GOLDEN_SET  # noqa: E402
from src.db import init_db  # noqa: E402
from src.graph import run_research  # noqa: E402


def main() -> None:
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        if len(sys.argv) < i + 3:
            print("Usage: python -m scripts.run_eval --compare <baseline> <candidate>")
            sys.exit(1)
        compare(sys.argv[i + 1], sys.argv[i + 2])
        return

    init_db()
    name = "latest"
    if "--name" in sys.argv:
        i = sys.argv.index("--name")
        name = sys.argv[i + 1]
    run_harness(GOLDEN_SET, run_fn=run_research, name=name)


if __name__ == "__main__":
    main()