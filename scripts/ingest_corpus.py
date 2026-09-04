"""CLI: index local documents, GitHub repositories, or YouTube transcripts.

Usage:
    python -m scripts.ingest_corpus <dir>
    python -m scripts.ingest_corpus --github <url>
    python -m scripts.ingest_corpus --youtube <video-id>
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import ingest_directory  # noqa: E402
from src.ingest_sources import (  # noqa: E402
    ingest_github_repository,
    ingest_youtube_transcript,
)


USAGE = (
    "Usage:\n"
    "  python -m scripts.ingest_corpus <directory>\n"
    "  python -m scripts.ingest_corpus --github <url>\n"
    "  python -m scripts.ingest_corpus --youtube <video-id>"
)


def _usage_error() -> None:
    print(USAGE, file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 1 and not args[0].startswith("--"):
        dir_path = Path(args[0])
        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}")
            raise SystemExit(1)
        total = ingest_directory(dir_path)
        source = str(dir_path)
    elif len(args) == 2 and args[0] == "--github":
        source = args[1]
        total = ingest_github_repository(source)
    elif len(args) == 2 and args[0] == "--youtube":
        source = args[1]
        total = ingest_youtube_transcript(source)
    else:
        _usage_error()
    print(f"\nIndexed {total} chunks from {source}")


if __name__ == "__main__":
    main()
