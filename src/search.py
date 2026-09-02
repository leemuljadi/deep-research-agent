"""Hybrid search using vector and full-text runs merged by weighted RRF."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from .config import settings
from .db import connect
from .llm import embed_text


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: int
    text: str
    score: float
    title: str | None = None
    url: str | None = None


RRF_K = 60  # standard RRF constant


def _rrf(
    runs: list[list[tuple[int, float]]], weights: list[float] | None = None
) -> dict[int, float]:
    """Merge ranked chunk lists by weighted Reciprocal Rank Fusion.

    Each run is a list of (chunk_id, score) already sorted by descending score.
    `weights` scales each run's contribution (default: equal weight). Returns
    {chunk_id: rrf_score}.
    """
    if weights is None:
        weights = [1.0] * len(runs)
    fused: dict[int, float] = {}
    for ranked, w in zip(runs, weights):
        for rank, (cid, _) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + w / (RRF_K + rank)
    return fused


def hybrid_search(query: str, top_k: int | None = None, vector_weight: float = 0.7) -> list[SearchResult]:
    top_k = top_k or settings.top_k
    query_vec = embed_text(query)

    with connect() as conn:
        with conn.cursor() as cur:
            # Vector run: cosine similarity.
            cur.execute(
                """
                SELECT c.id AS chunk_id, c.doc_id, c.content, d.title, d.url,
                       1 - (c.embedding <=> %s) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (query_vec, query_vec, top_k * 2),
            )
            vec_rows = cur.fetchall()

            # Full-text run.
            cur.execute(
                """
                SELECT c.id AS chunk_id, c.doc_id, c.content, d.title, d.url,
                       ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', %s)) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, top_k * 2),
            )
            fts_rows = cur.fetchall()

    vec_ranked = [(r["chunk_id"], float(r["score"])) for r in vec_rows]
    fts_ranked = [(r["chunk_id"], float(r["score"])) for r in fts_rows]

    fused = _rrf([vec_ranked, fts_ranked], weights=[vector_weight, 1.0 - vector_weight])

    # Build lookup for details.
    by_id: dict[int, dict] = {r["chunk_id"]: r for r in vec_rows}
    by_id.update({r["chunk_id"]: r for r in fts_rows})

    merged: list[SearchResult] = []
    for cid, rrf_score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        row = by_id[cid]
        merged.append(
            SearchResult(
                doc_id=row["doc_id"],
                chunk_id=cid,
                text=row["content"],
                score=round(rrf_score, 4),
                title=row["title"],
                url=row["url"],
            )
        )
    return merged


def get_document_text(doc_id: str) -> str | None:
    with psycopg.connect(settings.pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
            return row[0] if row else None
