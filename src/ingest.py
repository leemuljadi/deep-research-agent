"""Chunk documents, embed them, and index them in pgvector."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import settings
from .db import init_db, replace_document
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


def _sha1_16(identity: str) -> str:
    return hashlib.sha1(identity.encode()).hexdigest()[:16]


def _doc_id(path: Path, title: str) -> str:
    return _sha1_16(f"{path}:{title}")


def source_doc_id(namespace: str, identity: str) -> str:
    """Return a source-namespaced, stable SHA-1/16 document ID."""
    if not namespace or ":" in namespace:
        raise ValueError("source namespace must be non-empty and contain no colon")
    if not identity:
        raise ValueError("source identity must be non-empty")
    return _sha1_16(f"{namespace}:{identity}")


def ingest_text_document(
    *,
    doc_id: str,
    title: str,
    content: str,
    url: str | None,
) -> int:
    """Chunk, embed, and atomically replace one normalized text document."""
    chunks = chunk_text(content)
    embeddings = embed_texts(chunks)
    chunk_rows = list(zip(range(len(chunks)), chunks, embeddings, strict=True))
    replace_document(
        doc_id=doc_id,
        title=title,
        content=content,
        url=url,
        chunks=chunk_rows,
    )
    return len(chunks)


def ingest_file(path: Path, title: str | None = None) -> int:
    """Read a text/markdown file, chunk, embed and index it. Returns chunk count."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = title or path.stem
    return ingest_text_document(
        doc_id=_doc_id(path, title),
        title=title,
        content=text,
        url=str(path),
    )


def ingest_directory(dir_path: Path) -> int:
    """Index every .txt/.md/.py file under a directory. Returns total chunks."""
    init_db()
    total = 0
    for path in sorted(dir_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".py"}:
            try:
                total += ingest_file(path)
            except (OSError, UnicodeError) as exc:
                print(f"  [skip] {path.name}: {exc}")
            except AssertionError as exc:
                raise AssertionError(f"{path}: {exc}") from exc
            except TypeError as exc:
                raise TypeError(f"{path}: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Failed to ingest {path}: {exc}") from exc
    return total
