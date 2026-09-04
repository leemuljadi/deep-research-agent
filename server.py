"""HTTP API for the deep-research agent (VPS deployment).

Thin shell around the existing pipeline — no pipeline code is touched:
`init_db()` runs once at startup (idempotent), POST /research enqueues a
job row and returns its run id; the worker service executes the graph.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastmcp.utilities.lifespan import combine_lifespans
from pydantic import ValidationError

from src.db import (
    cancel_run,
    enqueue_job,
    get_job,
    init_db,
    record_decision,
    resume_run,
)
from src.config import settings
from src.mcp_server import mcp_app
from src.schemas import (
    GateDecision,
    ResearchRequest,
    RunAcceptedResponse,
    RunStatusResponse,
    RunTransitionResponse,
    job_row_to_status,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Idempotent: creates the pgvector extension, tables and HNSW/FTS indexes.
    init_db()
    if settings.run_cost_cap_warning:
        print(f"[api] WARNING: {settings.run_cost_cap_warning}")
    if settings.mcp_jwt_secret is None:
        print(
            "[api] WARNING: MCP_JWT_SECRET is unset; "
            "/mcp authentication is disabled (development only)"
        )
    yield


app = FastAPI(
    title="Deep-Research Agent",
    lifespan=combine_lifespans(lifespan, mcp_app.lifespan),
)


@app.get("/ping")
def ping() -> dict:
    return {"ok": True}


@app.post("/research", response_model=RunAcceptedResponse)
def research(body: ResearchRequest) -> dict:
    return RunAcceptedResponse(run_id=enqueue_job(body.question)).model_dump()


def _row_to_status(row: dict) -> RunStatusResponse:
    """Map one `research_jobs` row to its shared API/MCP shape."""
    try:
        return job_row_to_status(row)
    except ValidationError as ex:
        raise HTTPException(
            status_code=500,
            detail=f"stored report for run {row['id']} failed validation: {ex}",
        ) from ex


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def run_status(run_id: str) -> dict:
    row = get_job(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _row_to_status(row).model_dump()

_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the minimal research UI."""
    return FileResponse(_STATIC / "index.html")


@app.post(
    "/runs/{run_id}/approve",
    response_model=RunTransitionResponse,
    response_model_exclude_none=True,
)
def approve(run_id: str, decision: GateDecision | None = None) -> dict:
    """Approve the run waiting at a gate: re-enqueue from its snapshot (AD-14)."""
    if get_job(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    if not record_decision(run_id, "approve", (decision.payload if decision else {}) or {}):
        raise HTTPException(status_code=409, detail="run is not waiting_for_input")
    if not resume_run(run_id):
        # Lost the race between the decision record and the resume write.
        raise HTTPException(status_code=409, detail="run is not waiting_for_input")
    return {"run_id": run_id, "status": "queued"}


@app.post(
    "/runs/{run_id}/redirect",
    response_model=RunTransitionResponse,
    response_model_exclude_none=True,
)
def redirect(run_id: str, decision: GateDecision) -> dict:
    """Redirect the run at a gate: swap the question, then re-enqueue (AD-14).

    The api mutates the snapshot's question before re-enqueueing so the resumed
    run plans against the redirected question.
    """
    row = get_job(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    new_question = (decision.payload or {}).get("question")
    if not isinstance(new_question, str) or not new_question.strip():
        raise HTTPException(
            status_code=422, detail="redirect requires a non-empty 'question' payload"
        )
    if not record_decision(run_id, "redirect", dict(decision.payload)):
        raise HTTPException(status_code=409, detail="run is not waiting_for_input")
    if not resume_run(run_id, resume_question=new_question.strip()):
        raise HTTPException(status_code=409, detail="run is not waiting_for_input")
    return {"run_id": run_id, "status": "queued", "question": new_question.strip()}


@app.post(
    "/runs/{run_id}/cancel",
    response_model=RunTransitionResponse,
    response_model_exclude_none=True,
)
def cancel(run_id: str) -> dict:
    """Cancel a queued or waiting run; 409 when running (worker owns a claim)."""
    if get_job(run_id) is None:
        raise HTTPException(status_code=404, detail="run not found")
    if not cancel_run(run_id):
        raise HTTPException(status_code=409, detail="run is not cancellable")
    return {"run_id": run_id, "status": "cancelled"}


app.mount("/mcp", mcp_app)