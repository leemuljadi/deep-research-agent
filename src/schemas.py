"""Pydantic schemas for structured outputs and agent tool arguments."""
from __future__ import annotations

from pydantic import BaseModel, Field


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
