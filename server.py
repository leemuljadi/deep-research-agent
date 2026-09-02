"""HTTP API for the deep-research agent (VPS deployment).

Thin shell around the existing pipeline — no pipeline code is touched:
`init_db()` runs once at startup (idempotent), POST /research runs the
LangGraph plan -> research -> synthesise flow and returns the structured report.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.db import init_db
from src.graph import run_research


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Idempotent: creates the pgvector extension, tables and HNSW/FTS indexes.
    init_db()
    yield


app = FastAPI(title="Deep-Research Agent", lifespan=lifespan)


class ResearchIn(BaseModel):
    question: str


@app.get("/ping")
def ping() -> dict:
    return {"ok": True}


@app.post("/research")
def research(body: ResearchIn) -> dict:
    report = run_research(body.question)
    return report.model_dump()


_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the minimal research UI."""
    return FileResponse(_STATIC / "index.html")