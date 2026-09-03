"""LangGraph orchestration for the deep-research agent.

State machine: plan -> parallel researchers via ``Send`` fan-out -> synthesize.
The planner decomposes the question, one researcher runs per sub-question, and
the synthesizer aggregates their findings into a cited report.

Token usage and cost from every LLM call are accumulated in state and attached
to both the final report and tracing spans.

HITL gates (AD-14): the graph shape is unchanged by policy; the gated runner
executes nodes stepwise and stops at declared gate points, returning the gate
name so the worker can snapshot to Postgres and exit its claim. The pause
construct is one — ``run_research_state`` — while gate points are data.
"""
from __future__ import annotations

import json
import operator
import time
from dataclasses import asdict
from typing import Annotated, TypedDict

from pydantic import BaseModel, ValidationError

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents.planner import plan
from .agents.researcher import research_sub_question
from .agents.synthesizer import synthesize
from .llm import Usage
from .schemas import ResearchPlan, ResearchReport, SubQuestionResult
from .tracing import current_context, record_cost, trace


class ResearchState(TypedDict):
    question: str
    plan: ResearchPlan
    # Parallel researchers each return partial lists; `operator.add` merges them.
    sub_results: Annotated[list[SubQuestionResult], operator.add]
    usage_log: Annotated[list[Usage], operator.add]
    report: ResearchReport
    otel_ctx: object  # OTel context so researcher spans parent to the run span
    # HITL gate bookkeeping (AD-14): policy is per-run data, passed_gates marks
    # which gates have already been approved so a resumed run passes them.
    gate_policy: list[str]
    passed_gates: list[str]


GATE_POINTS: dict[str, str] = {
    # Gate point -> node after which the gate fires (pinned to graph shape):
    "plan": "planner",
    "synthesis": "synthesizer",
}
"""Fixed set of gate points (AD-14); per-run `gate_policy` picks from it."""


def initial_state(
    question: str,
    gates: list[str],
    *,
    passed_gates: list[str] | None = None,
) -> ResearchState:
    """Fresh run state with per-run gate policy attached (AD-14)."""
    return {
        "question": question,
        "plan": None,  # type: ignore[assignment]
        "sub_results": [],
        "usage_log": [],
        "report": None,  # type: ignore[assignment]
        "otel_ctx": current_context(),
        "gate_policy": list(gates),
        "passed_gates": list(passed_gates or []),
    }


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
        "sub_results": [
            SubQuestionResult(
                sub_question=payload["sub_question"],
                findings=findings,
                sources=sources,
            )
        ],
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
        initial = initial_state(question, [])  # sync path runs ungated (AD-14)
        final = graph.invoke(initial)
        report: ResearchReport = final["report"]
        record_cost(span, tokens=report.total_tokens, cost_usd=report.cost_usd)
    report.latency_seconds = round(time.time() - start, 3)
    return report


def run_research_state(
    state: ResearchState | None = None,
    gates: list[str] | None = None,
) -> tuple[ResearchState, str | None]:
    """Gated execution path (AD-14): run the graph stepwise, stop at a gate.

    Executes node by node. After each node, if that node hosts a gate point
    (per-run policy on ``state["gate_policy"]`` / ``gates``) that has not yet
    been passed, returns ``(state, gate_name)`` — the worker snapshots to
    Postgres and exits its claim. ``gate_name`` is None when the run ran to
    completion. A resumed invocation re-enters with the restored state; the
    already-passed gate is marked in ``passed_gates`` and does not re-fire.
    """
    if state is None:
        state = initial_state("", gates or [])
    state_start = time.time()
    policy = set(state.get("gate_policy") if gates is None else gates or [])
    passed = list(state.get("passed_gates") or [])

    # Re-entry: a resumed run continues after the node hosting its last passed
    # gate; a fresh run starts at the planner.
    steps = ["planner", "_researcher", "synthesizer"]
    entry_idx = 0
    for gate_name, node in GATE_POINTS.items():
        if gate_name in passed:
            entry_idx = max(entry_idx, steps.index(node) + 1)

    for node_name in steps[entry_idx:]:
        if node_name == "planner":
            _merge_update(state, _planner(state))
        elif node_name == "_researcher":
            # Send fan-out executed inline by the gated runner — one researcher
            # per planned sub-question; the graph shape is unchanged (AD-14).
            for q in state["plan"].sub_questions:
                _merge_update(
                    state,
                    _researcher(
                        {"sub_question": q, "otel_ctx": state.get("otel_ctx")}
                    ),
                )
        else:
            _merge_update(state, _synthesizer(state))

        # After the node, check whether a gate fires here (policy data decides;
        # already-passed gates never re-fire).
        gate = _gate_after(node_name, policy, passed)
        if gate is not None:
            state["passed_gates"] = passed + [gate]
            return state, gate

    # Run-level wall clock on completion: the report must carry latency even
    # though execution was segmented across gate pauses (wall clock includes
    # the waiting time — that IS the run's latency).
    report = state.get("report")
    if report is not None:
        report.latency_seconds = round(time.time() - state_start, 3)
    return state, None


def serialize_state(state: ResearchState) -> dict:
    """JSON-serializable snapshot of graph state for the Postgres round-trip.

    Pydantic models and Usage become dicts; otel_ctx is dropped — spans from a
    resumed segment parent to a fresh run span (AD-8 context is process-local).
    """
    out = dict(state)
    out.pop("otel_ctx", None)
    for key in ("plan", "report"):
        if isinstance(out.get(key), BaseModel):
            out[key] = json.loads(out[key].model_dump_json())
    out["sub_results"] = [
        json.loads(r.model_dump_json())
        for r in (out.get("sub_results") or [])
        if isinstance(r, BaseModel)
    ]
    out["usage_log"] = [asdict(u) for u in (out.get("usage_log") or [])]
    return out


def deserialize_state(snapshot: dict) -> ResearchState:
    """Rebuild graph state from its JSON snapshot (resume path, AD-14).

    Raises loudly on a corrupt snapshot — per spec, `fail_job`, never a silent
    degradation or a memory-resumed run.
    """
    try:
        state: ResearchState = {
            "question": snapshot["question"],
            "plan": None,
            "sub_results": [],
            "usage_log": [],
            "report": None,
            "otel_ctx": None,
            "gate_policy": _gate_list(snapshot, "gate_policy"),
            "passed_gates": _gate_list(snapshot, "passed_gates"),
        }
        if snapshot.get("plan") is not None:
            state["plan"] = ResearchPlan.model_validate(snapshot["plan"])
        for r in snapshot.get("sub_results") or []:
            state["sub_results"].append(SubQuestionResult.model_validate(r))
        for u in snapshot.get("usage_log") or []:
            state["usage_log"].append(Usage(**u))
        if snapshot.get("report") is not None:
            state["report"] = ResearchReport.model_validate(snapshot["report"])
        # Consistency: a snapshot may not claim a gate was passed whose node
        # output is missing — that combination crashes the resumed run later
        # with a confusing AttributeError instead of failing loudly here.
        if "plan" in state["passed_gates"] and state["plan"] is None:
            raise ValueError("corrupt state snapshot: 'plan' gate passed but plan is missing")
        return state
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise ValueError(f"corrupt state snapshot: {exc}") from exc


def _gate_list(snapshot: dict, key: str) -> list[str]:
    """Extract a gate list from a snapshot, rejecting non-list/string corruption.

    A string like \"plan\" would list()-split into ['p','l','a','n'] and match
    no gate point — silently bypassing every gate. Corrupt data raises.
    """
    value = snapshot.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(g, str) for g in value):
        raise ValueError(f"{key} must be a list of gate names, got {value!r}")
    unknown = [g for g in value if g not in GATE_POINTS]
    if unknown:
        raise ValueError(f"{key} contains unknown gate names: {unknown!r}")
    return list(value)


def _merge_update(state: ResearchState, update: dict) -> None:
    """Fold one node update into the working state (reducer semantics)."""
    for key, value in update.items():
        if isinstance(value, list) and isinstance(state.get(key), list):
            state[key] = state[key] + value
        else:
            state[key] = value


def _gate_after(
    node_name: str, policy: set[str], passed: list[str]
) -> str | None:
    """The gate point hosted by this node, if it fires per policy and not yet passed."""
    for gate_name, node in GATE_POINTS.items():
        if node == node_name and gate_name in policy and gate_name not in passed:
            return gate_name
    return None