"""CLI: run the deep-research agent on a question.

Usage:
    python -m scripts.run_research "your question here"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import init_db  # noqa: E402
from src.graph import run_research  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.run_research \"your question\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    init_db()
    report = run_research(question)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(report.summary)
    print("\nFINDINGS")
    for i, f in enumerate(report.findings, 1):
        print(f"  {i}. {f}")
    print("\nSOURCES")
    for s in report.sources:
        print(f"  - {s.title} ({s.url or 'no url'})")
    print("\n" + "=" * 70)
    print(f"confidence={report.confidence:.2f}  "
          f"cost=${report.cost_usd:.4f}  "
          f"latency={report.latency_seconds:.2f}s")


if __name__ == "__main__":
    main()
