"""Chunk documents, embed them, and index them in pgvector."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import settings
from .db import init_db, insert_chunks, upsert_document
from .llm import embed_texts


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks on paragraph/word boundaries."""
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    # Normalise whitespace but keep paragraph breaks as chunk seeds.
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # Split long paragraphs.
            while len(para) > size:
                chunks.append(para[:size])
                para = para[size - overlap :]
            current = para
    if current:
        chunks.append(current)
    return chunks or [text]


def _doc_id(path: Path, title: str) -> str:
    raw = f"{path}:{title}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def ingest_file(path: Path, title: str | None = None) -> int:
    """Read a text/markdown file, chunk, embed and index it. Returns chunk count."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = title or path.stem
    doc_id = _doc_id(path, title)

    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    upsert_document(doc_id=doc_id, title=title, content=text, url=str(path))
    insert_chunks(doc_id, list(enumerate(zip(chunks, embeddings))))
    return len(chunks)


def ingest_directory(dir_path: Path) -> int:
    """Index every .txt/.md/.py file under a directory. Returns total chunks."""
    init_db()
    total = 0
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".py"}:
            try:
                total += ingest_file(path)
            except Exception as exc:  # noqa: BLE001 - keep indexing the rest
                print(f"  [skip] {path.name}: {exc}")
    return total
