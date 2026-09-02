"""Postgres + pgvector storage for schema management and corpus writes.

This module owns DDL and write queries. ``search.py`` owns the retrieval
contract's read queries.
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from .config import settings


def connect() -> psycopg.Connection[dict[str, Any]]:
    conn = psycopg.connect(settings.pg_dsn, row_factory=dict_row)
    register_vector(conn)  # enables vector type handling
    return conn


def init_db() -> None:
    """Create the pgvector extension and schema (idempotent)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    url         TEXT,
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          BIGSERIAL PRIMARY KEY,
                    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector({settings.embedding_dim})
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
                ON chunks USING hnsw (embedding vector_cosine_ops)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_content_fts
                ON chunks USING gin (to_tsvector('english', content))
                """
            )
        conn.commit()


def upsert_document(doc_id: str, title: str, content: str, url: str | None = None) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, title, url, content)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    url = EXCLUDED.url,
                    content = EXCLUDED.content
                """,
                (doc_id, title, url, content),
            )
        conn.commit()


def insert_chunks(doc_id: str, chunks: list[tuple[int, str, list[float]]]) -> None:
    """Insert (chunk_index, text, embedding) tuples for a document."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            for idx, text, emb in chunks:
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (doc_id, idx, text, emb),
                )
        conn.commit()


def delete_document(doc_id: str) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
