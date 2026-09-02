"""CLI: index a directory of documents into pgvector.

Usage:
    python -m scripts.ingest_corpus <dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import ingest_directory  # noqa: E402
from src.db import init_db  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.ingest_corpus <directory>")
        sys.exit(1)
    dir_path = Path(sys.argv[1])
    if not dir_path.is_dir():
        print(f"Not a directory: {dir_path}")
        sys.exit(1)
    init_db()
    total = ingest_directory(dir_path)
    print(f"\nIndexed {total} chunks from {dir_path}")


if __name__ == "__main__":
    main()
