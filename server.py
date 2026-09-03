"""HTTP API for the deep-research agent (VPS deployment).

Thin shell around the existing pipeline — no pipeline code is touched:
`init_db()` runs once at startup (idempotent), POST /research enqueues a
job row and returns its run id; the worker service executes the graph.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, StringConstraints

from src.db import enqueue_job, init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Idempotent: creates the pgvector extension, tables and HNSW/FTS indexes.
    init_db()
    yield


app = FastAPI(title="Deep-Research Agent", lifespan=lifespan)


class ResearchIn(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@app.get("/ping")
def ping() -> dict:
    return {"ok": True}


@app.post("/research")
def research(body: ResearchIn) -> dict:
    job_id = enqueue_job(body.question)
    return {"run_id": job_id}


_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the minimal research UI."""
    return FileResponse(_STATIC / "index.html")