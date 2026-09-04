"""FastMCP exposure of the asynchronous research job boundary (AD-17)."""
from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.providers.jwt import JWTVerifier
from pydantic import ValidationError

from src import db
from src.config import settings
from src.schemas import (
    NonEmptyText,
    RunAcceptedResponse,
    RunStatus,
    RunStatusResponse,
    RunTransitionResponse,
    job_row_to_status,
)


def _jwt_verifier(secret: str) -> JWTVerifier:
    """Build the private-deployment HS256 verifier required by AD-17."""
    return JWTVerifier(public_key=secret, algorithm="HS256")


def build_mcp_server(jwt_secret: str | None) -> FastMCP:
    """Build a server exposing exactly the five asynchronous run verbs."""
    auth = _jwt_verifier(jwt_secret) if jwt_secret else None
    server = FastMCP("Deep-Research Agent", auth=auth)

    @server.tool
    def submit(question: NonEmptyText) -> RunAcceptedResponse:
        """Enqueue a research run and return immediately with its identifier."""
        return RunAcceptedResponse(run_id=db.enqueue_job(question))

    @server.tool
    def poll(run_id: str) -> RunStatusResponse:
        """Read the current status and terminal result of a research run."""
        row = db.get_job(run_id)
        if row is None:
            raise ToolError("run not found")
        try:
            return job_row_to_status(row)
        except (ValidationError, ValueError) as ex:
            raise ToolError(f"stored result for run {run_id} failed validation") from ex

    @server.tool
    def approve(run_id: str) -> RunTransitionResponse:
        """Approve a run waiting at a human gate and re-enqueue it."""
        if db.get_job(run_id) is None:
            raise ToolError("run not found")
        if not db.record_decision(run_id, "approve", {}):
            raise ToolError("run is not waiting_for_input")
        if not db.resume_run(run_id):
            raise ToolError("run is not waiting_for_input")
        return RunTransitionResponse(run_id=run_id, status=RunStatus.QUEUED)

    @server.tool
    def redirect(run_id: str, question: NonEmptyText) -> RunTransitionResponse:
        """Replace the question at a human gate and re-enqueue the run."""
        if db.get_job(run_id) is None:
            raise ToolError("run not found")
        payload = {"question": question}
        if not db.record_decision(run_id, "redirect", payload):
            raise ToolError("run is not waiting_for_input")
        if not db.resume_run(run_id, resume_question=question):
            raise ToolError("run is not waiting_for_input")
        return RunTransitionResponse(
            run_id=run_id,
            status=RunStatus.QUEUED,
            question=question,
        )

    @server.tool
    def cancel(run_id: str) -> RunTransitionResponse:
        """Cancel a queued or waiting research run."""
        if db.get_job(run_id) is None:
            raise ToolError("run not found")
        if not db.cancel_run(run_id):
            raise ToolError("run is not cancellable")
        return RunTransitionResponse(run_id=run_id, status=RunStatus.CANCELLED)

    return server


mcp = build_mcp_server(settings.mcp_jwt_secret)
mcp_app = mcp.http_app(path="/", stateless_http=True)
