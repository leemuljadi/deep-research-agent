"""Postgres + pgvector storage for schema management and corpus writes.

This module owns DDL and write queries. ``search.py`` owns the retrieval
contract's read queries.
"""
from __future__ import annotations

import uuid
import psycopg
from typing import Any
from psycopg.types.json import Json
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
                CREATE TABLE IF NOT EXISTS research_jobs (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    question    TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'queued'
                                CHECK (status IN (
                                    'queued', 'running', 'waiting_for_input',
                                    'completed', 'failed', 'cancelled',
                                    'cost_cap_exceeded'
                                )),
                    worker_id   TEXT,
                    result      JSONB,
                    error       TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS research_jobs_status_created
                ON research_jobs (status, created_at)
                """
            )
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


def enqueue_job(question: str) -> str:
    """Insert a queued job row and return its id (UUID hex)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_jobs (question)
                VALUES (%s)
                RETURNING id
                """,
                (question,),
            )
            row = cur.fetchone()
            if row is None:  # defensive: INSERT..RETURNING always yields a row
                raise RuntimeError("enqueue_job: INSERT RETURNING produced no row")
            job_id = row["id"]
        conn.commit()
    return str(job_id)


# Single-statement atomic claim (AD-14): the row lock inside the claiming
# transaction guarantees exactly one winner when workers race.
_CLAIM_SQL = (
    "UPDATE research_jobs "
    "SET status = 'running', worker_id = %s, updated_at = now() "
    "WHERE id = ("
    "  SELECT id FROM research_jobs"
    "  WHERE status = 'queued'"
    "  ORDER BY created_at"
    "  FOR UPDATE SKIP LOCKED LIMIT 1"
    ") RETURNING *"
)


def claim_next_job(worker_id: str) -> dict[str, Any] | None:
    """Atomically claim the oldest queued job, or return None on empty queue."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_CLAIM_SQL, (worker_id,))
            row = cur.fetchone()
        conn.commit()
    return row


def complete_job(job_id: str, report: dict[str, Any]) -> None:
    """Mark a job completed and store its report JSON."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE research_jobs
                SET status = 'completed', result = %s, updated_at = now()
                WHERE id = %s
                """,
                (Json(report), job_id),
            )
        conn.commit()


def fail_job(job_id: str, error: str) -> None:
    """Mark a job failed with its error message recorded."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE research_jobs
                SET status = 'failed', error = %s, updated_at = now()
                WHERE id = %s
                """,
                (error, job_id),
            )
        conn.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    """Read one job row by id (None when absent or not a valid UUID)."""
    try:
        uuid.UUID(job_id)
    except (ValueError, AttributeError):
        return None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM research_jobs WHERE id = %s",
                (job_id,),
            )
            row = cur.fetchone()
    return row
