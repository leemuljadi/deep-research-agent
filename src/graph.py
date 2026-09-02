"""LangGraph orchestration for the deep-research agent.

State machine: plan -> parallel researchers via ``Send`` fan-out -> synthesize.
The planner decomposes the question, one researcher runs per sub-question, and
the synthesizer aggregates their findings into a cited report.

Token usage and cost from every LLM call are accumulated in state and attached
to both the final report and tracing spans.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.planner import plan
from .agents.researcher import research_sub_question
from .agents.synthesizer import synthesize
from .llm import Usage
from .schemas import ResearchPlan, ResearchReport
from .tracing import current_context, record_cost, trace


class ResearchState(TypedDict):
    question: str
    plan: ResearchPlan
    # Parallel researchers each return partial lists; `operator.add` merges them.
    sub_results: Annotated[list, operator.add]  # list of (sub_question, findings, sources)
    usage_log: Annotated[list, operator.add]  # list[Usage] from every LLM call
    report: ResearchReport
    otel_ctx: object  # OTel context so researcher spans parent to the run span


def _planner(state: ResearchState) -> dict:
    with trace("plan") as span:
        p, usage = plan(state["question"])
        record_cost(span, tokens=usage.total_tokens, cost_usd=usage.cost_usd)
    return {"plan": p, "usage_log": [usage]}


def _fan_out_researchers(state: ResearchState) -> list[Send]:
    """Fan out one parallel researcher sub-agent per planned sub-question."""
    return [
        Send("_researcher", {"sub_question": q, "otel_ctx": state.get("otel_ctx")})
        for q in state["plan"].sub_questions
    ]


def _researcher(payload: dict) -> dict:
    """One researcher sub-agent: retrieve + read sources for its sub-question."""
    with trace(
        "research", ctx=payload.get("otel_ctx"), sub_question=payload["sub_question"]
    ) as span:
        findings, sources, usage = research_sub_question(payload["sub_question"])
        record_cost(span, tokens=usage.total_tokens, cost_usd=usage.cost_usd)
    return {
        "sub_results": [(payload["sub_question"], findings, sources)],
        "usage_log": [usage],
    }


def _synthesizer(state: ResearchState) -> dict:
    with trace("synthesize", ctx=state.get("otel_ctx")) as span:
        report, usage = synthesize(state["plan"], state["sub_results"])
        prior = state.get("usage_log") or []
        report.total_tokens = sum(u.total_tokens for u in prior) + usage.total_tokens
        report.cost_usd = round(sum(u.cost_usd for u in prior) + usage.cost_usd, 6)
        record_cost(span, tokens=report.total_tokens, cost_usd=report.cost_usd)
    return {"report": report, "usage_log": [usage]}


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", _planner)
    g.add_node("_researcher", _researcher)
    g.add_node("synthesizer", _synthesizer)
    g.add_edge(START, "planner")
    # Map-reduce: planner fans out one Send per sub-question; all researcher
    # instances run concurrently and reduce into `sub_results`/`usage_log`.
    g.add_conditional_edges("planner", _fan_out_researchers, ["_researcher"])
    g.add_edge("_researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def run_research(question: str) -> ResearchReport:
    """End-to-end research run: plan -> parallel researchers -> synthesise."""
    start = time.time()
    graph = build_graph()
    with trace("research_run", question=question[:200]) as span:
        initial: ResearchState = {
            "question": question,
            "plan": None,  # type: ignore[assignment]
            "sub_results": [],
            "usage_log": [],
            "report": None,  # type: ignore[assignment]
            "otel_ctx": current_context(),
        }
        final = graph.invoke(initial)
        report: ResearchReport = final["report"]
        record_cost(span, tokens=report.total_tokens, cost_usd=report.cost_usd)
    report.latency_seconds = round(time.time() - start, 3)
    return report