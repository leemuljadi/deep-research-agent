"""Pydantic schemas for structured outputs and agent tool arguments."""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, Field, StringConstraints


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


class PlannerReflection(BaseModel):
    """Validated graph-owned review of one completed research round."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality/completeness score for the current plan and findings.",
    )
    needs_more_research: bool = Field(
        description="Whether meaningful unresolved gaps remain."
    )
    additional_sub_questions: list[str] = Field(
        default_factory=list,
        description="Novel questions that would close the remaining gaps.",
    )
    rationale: str = Field(description="Brief reason for the score and decision.")


class SynthesisReview(BaseModel):
    """Validated graph-owned review of a draft against the research findings."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Grounded completeness score for the current draft.",
    )
    needs_revision: bool = Field(
        description="Whether the draft has a meaningful correctable omission."
    )
    feedback: str = Field(
        description="Specific revision guidance grounded in the supplied findings."
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


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ResearchRequest(BaseModel):
    """A validated research-job submission."""

    question: NonEmptyText


class RunAcceptedResponse(BaseModel):
    """Identifier returned immediately after a job is enqueued."""

    run_id: str


class RunTransitionResponse(BaseModel):
    """Result of an API/MCP mutation at the run boundary."""

    run_id: str
    status: RunStatus
    question: str | None = None


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


def job_row_to_status(row: Mapping[str, Any]) -> RunStatusResponse:
    """Validate one database job row into the shared status response."""
    status = RunStatus(row["status"])
    report = None
    if status == RunStatus.COMPLETED and row.get("result") is not None:
        report = ResearchReport.model_validate(row["result"])
    return RunStatusResponse(
        run_id=str(row["id"]),
        status=status,
        question=row["question"],
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
        report=report,
        error=row.get("error")
        if status in (RunStatus.FAILED, RunStatus.COST_CAP_EXCEEDED)
        else None,
    )


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


class ToolResult(BaseModel):
    """Terminal result of one tool execution (AD-16).

    Every tool call through the executor ends in exactly one named state:
    ok, not_found, disabled, error, or timeout — never an open-ended await.
    """

    tool: str = Field(description="Tool name that produced this result.")
    state: Literal["ok", "not_found", "disabled", "error", "timeout"]
    payload: Any = None
    detail: str | None = None
