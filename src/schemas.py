"""Pydantic schemas for structured outputs and agent tool arguments."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from enum import StrEnum


class RunStatus(StrEnum):
    """Closed run/job status set (AD-14): terminal = completed/failed/cancelled/cost_cap_exceeded."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COST_CAP_EXCEEDED = "cost_cap_exceeded"


# --- HITL gate payloads (AD-6, AD-14) -----------------------------------------

class GateDecision(BaseModel):
    """A human decision taken at a gate: approve, or redirect with a new question."""

    decision: Literal["approve", "redirect"] = Field(
        description="Decision taken at the gate."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Decision payload; redirect carries {'question': ...}.",
    )


# --- Structured outputs -------------------------------------------------------

class ResearchPlan(BaseModel):
    """The lead agent's decomposition of a question into sub-questions."""

    objective: str = Field(description="Restated research objective.")
    sub_questions: list[str] = Field(
        description="Independent sub-questions a researcher agent should answer."
    )


class Source(BaseModel):
    """A retrieved / grounded source."""

    title: str
    url: str | None = None
    snippet: str
    score: float | None = None


class SubQuestionResult(BaseModel):
    """Validated findings and sources for one researched sub-question."""

    sub_question: str
    findings: list[str]
    sources: list[Source]


# --- Run / job status shapes --------------------------------------------------

class RunStatusResponse(BaseModel):
    """Run status shape served by the API and the worker's terminal writes."""

    run_id: str = Field(description="Run/job identifier (UUID).")
    status: RunStatus = Field(description="Closed run status enum.")
    question: str = Field(description="The submitted research question.")
    created_at: str = Field(description="Job creation timestamp (ISO 8601).")
    updated_at: str = Field(description="Last status transition timestamp (ISO 8601).")
    report: ResearchReport | None = Field(
        default=None, description="Validated report when completed, else None."
    )
    error: str | None = Field(
        default=None,
        description="Error/summary message when failed or cost_cap_exceeded, else None.",
    )


class ResearchReport(BaseModel):
    """The final grounded, structured report."""

    summary: str = Field(description="Executive summary answering the question.")
    findings: list[str] = Field(description="Key findings, each grounded in sources.")
    sources: list[Source] = Field(description="Sources that ground the findings.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence.")
    cost_usd: float = Field(default=0.0, description="Approx token cost of the run.")
    latency_seconds: float = Field(default=0.0, description="End-to-end run time.")
    total_tokens: int = Field(default=0, description="Total tokens consumed by the run.")


# --- Agent tool schemas -------------------------------------------------------

class RetrieveToolInput(BaseModel):
    """Arguments for the `retrieve` tool."""

    query: str = Field(description="Search query over the indexed corpus.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return.")


class ReadDocumentToolInput(BaseModel):
    """Arguments for the `read_document` tool."""

    doc_id: str = Field(description="Document identifier in the store.")


class WebSearchToolInput(BaseModel):
    """Arguments for the `web_search` tool (stub / pluggable)."""

    query: str = Field(description="Web search query.")
    max_results: int = Field(default=5, ge=1, le=10)
