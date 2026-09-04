"""LangGraph orchestration for the deep-research agent.

The graph owns the two bounded reflection transitions sanctioned by AD-7:
planner re-planning between research rounds and draft review before completion.
Agent modules stay single-call functions; reflection, iteration bounds, candidate
retention, HITL placement, and snapshot/resume all live here.
"""
from __future__ import annotations

import json
import math
import operator
import time
from dataclasses import asdict
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ValidationError

from .agents.planner import plan
from .agents.researcher import research_sub_question
from .agents.synthesizer import synthesize
from .config import settings
from .llm import Usage, chat_with_usage
from .schemas import (
    PlannerReflection,
    ResearchPlan,
    ResearchReport,
    SubQuestionResult,
    SynthesisReview,
)
from .tracing import current_context, record_cost, trace

SCORE_EPSILON = 1e-6

_PLANNER_REFLECTION_SYSTEM = (
    "You review a completed research round for meaningful unresolved gaps. "
    "Score the current plan plus findings from 0 to 1. Request another round "
    "only when specific additional research can materially improve the answer. "
    "Every additional question must be novel: do not repeat or paraphrase a "
    "question already in the plan. Return ONLY valid JSON matching: "
    '{"score": float, "needs_more_research": bool, '
    '"additional_sub_questions": [str, ...], "rationale": str}.'
)

_SYNTHESIS_REVIEW_SYSTEM = (
    "Review a draft research report against the supplied plan and findings. "
    "Score grounded completeness from 0 to 1. Request revision only for a "
    "specific, meaningful omission or unsupported statement that the supplied "
    "findings can correct. Return ONLY valid JSON matching: "
    '{"score": float, "needs_revision": bool, "feedback": str}.'
)


class ResearchState(TypedDict):
    question: str
    plan: ResearchPlan
    sub_results: Annotated[list[SubQuestionResult], operator.add]
    usage_log: Annotated[list[Usage], operator.add]
    report: ResearchReport
    otel_ctx: object
    gate_policy: list[str]
    passed_gates: list[str]
    reflection_enabled: bool
    planner_reflection_max_iterations: int
    synthesis_review_max_iterations: int
    planner_iterations: int
    pending_sub_questions: list[str]
    planner_reflection: PlannerReflection | None
    planner_complete: bool
    best_plan: ResearchPlan | None
    best_plan_score: float | None
    best_sub_results: list[SubQuestionResult]
    synthesis_iterations: int
    synthesis_review: SynthesisReview | None
    synthesis_feedback: str | None
    synthesis_complete: bool
    best_report: ResearchReport | None
    best_report_score: float | None


GATE_POINTS: dict[str, str] = {
    "plan": "planner",
    "synthesis": "synthesis_review",
}
"""Fixed gate names; per-run ``gate_policy`` selects from this set."""


def initial_state(
    question: str,
    gates: list[str],
    *,
    passed_gates: list[str] | None = None,
    reflection_enabled: bool | None = None,
    planner_reflection_max_iterations: int | None = None,
    synthesis_review_max_iterations: int | None = None,
) -> ResearchState:
    """Create fresh state and freeze reflection defaults for the whole run."""
    enabled = settings.reflection_enabled if reflection_enabled is None else reflection_enabled
    planner_cap = (
        settings.planner_reflection_max_iterations
        if planner_reflection_max_iterations is None
        else planner_reflection_max_iterations
    )
    synthesis_cap = (
        settings.synthesis_review_max_iterations
        if synthesis_review_max_iterations is None
        else synthesis_review_max_iterations
    )
    if not isinstance(enabled, bool):
        raise ValueError("reflection_enabled must be a boolean")
    planner_cap = _non_negative_int(planner_cap, "planner_reflection_max_iterations")
    synthesis_cap = _non_negative_int(synthesis_cap, "synthesis_review_max_iterations")
    return {
        "question": question,
        "plan": None,  # type: ignore[assignment]
        "sub_results": [],
        "usage_log": [],
        "report": None,  # type: ignore[assignment]
        "otel_ctx": current_context(),
        "gate_policy": list(gates),
        "passed_gates": list(passed_gates or []),
        "reflection_enabled": enabled,
        "planner_reflection_max_iterations": planner_cap,
        "synthesis_review_max_iterations": synthesis_cap,
        "planner_iterations": 0,
        "pending_sub_questions": [],
        "planner_reflection": None,
        "planner_complete": False,
        "best_plan": None,
        "best_plan_score": None,
        "best_sub_results": [],
        "synthesis_iterations": 0,
        "synthesis_review": None,
        "synthesis_feedback": None,
        "synthesis_complete": False,
        "best_report": None,
        "best_report_score": None,
    }


def _planner(state: ResearchState) -> dict:
    with trace("plan") as span:
        candidate, usage = plan(state["question"])
        record_cost(span, tokens=usage.total_tokens, cost_usd=usage.cost_usd)
    return {
        "plan": candidate,
        "pending_sub_questions": list(candidate.sub_questions),
        "usage_log": [usage],
    }


def _fan_out_researchers(state: ResearchState) -> list[Send]:
    """Fan out only questions selected for the current research round."""
    return [
        Send("_researcher", {"sub_question": q, "otel_ctx": state.get("otel_ctx")})
        for q in state.get("pending_sub_questions") or state["plan"].sub_questions
    ]


def _researcher(payload: dict) -> dict:
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


def _planner_reflection(state: ResearchState) -> dict:
    """Score one completed round and either select novel work or exit boundedly."""
    if not state["reflection_enabled"] or state["planner_reflection_max_iterations"] == 0:
        return _finish_planner_without_reflection(state)

    payload = {
        "question": state["question"],
        "plan": state["plan"].model_dump(),
        "findings": [result.model_dump() for result in state["sub_results"]],
    }
    with trace("planner_reflection", ctx=state.get("otel_ctx")) as span:
        raw, usage = chat_with_usage(
            [
                {"role": "system", "content": _PLANNER_REFLECTION_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.0,
        )
        reflection = PlannerReflection.model_validate_json(_strip_fence(raw))
        record_cost(span, tokens=usage.total_tokens, cost_usd=usage.cost_usd)

    update = _apply_planner_reflection(state, reflection)
    update["planner_reflection"] = reflection
    update["usage_log"] = [usage]
    return update


def _finish_planner_without_reflection(state: ResearchState) -> dict:
    plan_candidate = state["plan"].model_copy(deep=True)
    results_candidate = _copy_results(state["sub_results"])
    return {
        "plan": plan_candidate,
        "best_plan": plan_candidate.model_copy(deep=True),
        "best_sub_results": results_candidate,
        "pending_sub_questions": [],
        "planner_complete": True,
    }


def _apply_planner_reflection(state: ResearchState, reflection: PlannerReflection) -> dict:
    prior_best_score = state.get("best_plan_score")
    improved = prior_best_score is None or reflection.score > prior_best_score + SCORE_EPSILON
    if improved:
        best_plan = state["plan"].model_copy(deep=True)
        best_results = _copy_results(state["sub_results"])
        best_score = reflection.score
    else:
        best_plan = state["best_plan"].model_copy(deep=True)  # type: ignore[union-attr]
        best_results = _copy_results(state["best_sub_results"])
        best_score = prior_best_score

    stop = {
        "plan": best_plan.model_copy(deep=True),
        "best_plan": best_plan,
        "best_plan_score": best_score,
        "best_sub_results": best_results,
        "pending_sub_questions": [],
        "planner_complete": True,
    }
    if not improved:
        return stop
    if state["planner_iterations"] >= state["planner_reflection_max_iterations"]:
        return stop
    if not reflection.needs_more_research:
        return stop

    novel = [question.strip() for question in reflection.additional_sub_questions]
    known = {_normalize_question(question) for question in state["plan"].sub_questions}
    normalized = [_normalize_question(question) for question in novel]
    if (
        not novel
        or any(not question for question in novel)
        or any(question in known for question in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        return stop

    candidate = ResearchPlan(
        objective=state["plan"].objective,
        sub_questions=[*state["plan"].sub_questions, *novel],
    )
    return {
        "plan": candidate,
        "best_plan": best_plan,
        "best_plan_score": best_score,
        "best_sub_results": best_results,
        "pending_sub_questions": novel,
        "planner_iterations": state["planner_iterations"] + 1,
        "planner_complete": False,
    }


def _synthesizer(state: ResearchState) -> dict:
    """Create or revise one draft; the review node decides whether it is final."""

    plan_candidate = state.get("best_plan") or state["plan"]
    results = _copy_results(state.get("best_sub_results") or state["sub_results"])
    feedback = state.get("synthesis_feedback")
    if feedback:
        results.append(_revision_context(state["report"].model_copy(deep=True), feedback))

    with trace("synthesize", ctx=state.get("otel_ctx")) as span:
        report, usage = synthesize(plan_candidate, results)
        _fold_usage_into_report(report, [*(state.get("usage_log") or []), usage])
        record_cost(span, tokens=report.total_tokens, cost_usd=report.cost_usd)

    return {"report": report, "usage_log": [usage]}


def _synthesis_review(state: ResearchState) -> dict:
    """Review a draft and select the best-scoring report under the revision cap."""
    if not state["reflection_enabled"] or state["synthesis_review_max_iterations"] == 0:
        report = state["report"].model_copy(deep=True)
        return {
            "best_report": report,
            "report": report.model_copy(deep=True),
            "synthesis_complete": True,
            "synthesis_feedback": None,
        }

    plan_candidate = state.get("best_plan") or state["plan"]
    results = state.get("best_sub_results") or state["sub_results"]
    payload = {
        "plan": plan_candidate.model_dump(),
        "findings": [result.model_dump() for result in results],
        "draft": state["report"].model_dump(
            exclude={"total_tokens", "cost_usd", "latency_seconds"}
        ),
    }
    with trace("synthesis_review", ctx=state.get("otel_ctx")) as span:
        raw, usage = chat_with_usage(
            [
                {"role": "system", "content": _SYNTHESIS_REVIEW_SYSTEM},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.0,
        )
        review = SynthesisReview.model_validate_json(_strip_fence(raw))
        record_cost(span, tokens=usage.total_tokens, cost_usd=usage.cost_usd)

    update = _apply_synthesis_review(state, review)
    complete_usage = [*(state.get("usage_log") or []), usage]
    for key in ("report", "best_report"):
        candidate = update.get(key)
        if candidate is not None:
            _fold_usage_into_report(candidate, complete_usage)
    update["synthesis_review"] = review
    update["usage_log"] = [usage]
    return update


def _apply_synthesis_review(state: ResearchState, review: SynthesisReview) -> dict:
    prior_best_score = state.get("best_report_score")
    improved = prior_best_score is None or review.score > prior_best_score + SCORE_EPSILON
    if improved:
        best_report = state["report"].model_copy(deep=True)
        best_score = review.score
    else:
        best_report = state["best_report"].model_copy(deep=True)  # type: ignore[union-attr]
        best_score = prior_best_score

    stop = {
        "report": best_report.model_copy(deep=True),
        "best_report": best_report,
        "best_report_score": best_score,
        "synthesis_feedback": None,
        "synthesis_complete": True,
    }
    if not improved:
        return stop
    if state["synthesis_iterations"] >= state["synthesis_review_max_iterations"]:
        return stop
    if not review.needs_revision or not review.feedback.strip():
        return stop
    return {
        "best_report": best_report,
        "best_report_score": best_score,
        "synthesis_feedback": review.feedback.strip(),
        "synthesis_iterations": state["synthesis_iterations"] + 1,
        "synthesis_complete": False,
    }


def _revision_context(report: ResearchReport, feedback: str) -> SubQuestionResult:
    return SubQuestionResult(
        sub_question="Synthesis review of the prior draft",
        findings=[
            "PRIOR DRAFT:\n"
            + report.model_dump_json(
                exclude={"total_tokens", "cost_usd", "latency_seconds"}
            ),
            f"REVIEW FEEDBACK:\n{feedback}",
        ],
        sources=[source.model_copy(deep=True) for source in report.sources],
    )


def _fold_usage_into_report(report: ResearchReport, usage_log: list[Usage]) -> None:
    """The synthesizer node remains the sole mutator of report usage totals."""
    report.total_tokens = sum(usage.total_tokens for usage in usage_log)
    report.cost_usd = round(sum(usage.cost_usd for usage in usage_log), 6)


def _route_after_planner_reflection(state: ResearchState):
    if state["planner_complete"]:
        return "synthesizer"
    return _fan_out_researchers(state)


def _route_after_synthesis_review(state: ResearchState):
    return END if state["synthesis_complete"] else "synthesizer"


def build_graph():
    """Compile the fixed map-reduce graph with exactly two reflection edges."""
    graph = StateGraph(ResearchState)
    graph.add_node("planner", _planner)
    graph.add_node("_researcher", _researcher)
    graph.add_node("planner_reflection", _planner_reflection)
    graph.add_node("synthesizer", _synthesizer)
    graph.add_node("synthesis_review", _synthesis_review)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", _fan_out_researchers, ["_researcher"])
    graph.add_edge("_researcher", "planner_reflection")
    graph.add_conditional_edges(
        "planner_reflection",
        _route_after_planner_reflection,
        ["_researcher", "synthesizer"],
    )
    graph.add_edge("synthesizer", "synthesis_review")
    graph.add_conditional_edges(
        "synthesis_review",
        _route_after_synthesis_review,
        ["synthesizer", END],
    )
    return graph.compile()


def run_research(question: str) -> ResearchReport:
    """End-to-end execution through the same bounded runner used by workers."""
    start = time.time()
    with trace("research_run", question=question[:200]) as span:
        state, gate = run_research_state(initial_state(question, []))
        if gate is not None:
            raise RuntimeError(f"unexpected gate in ungated run: {gate}")
        report = state["report"]
        record_cost(span, tokens=report.total_tokens, cost_usd=report.cost_usd)
    report.latency_seconds = round(time.time() - start, 3)
    return report


def run_research_state(
    state: ResearchState | None = None,
    gates: list[str] | None = None,
) -> tuple[ResearchState, str | None]:
    """Run until completion or the next declared HITL gate.

    Synthesis reflection finishes before the synthesis gate can fire, so its
    snapshot already holds the selected best report. Resuming from that gate
    returns without another agent or reflection call.
    """
    if state is None:
        state = initial_state("", gates or [])
    started = time.time()
    policy = set(state.get("gate_policy") if gates is None else gates or [])
    passed = list(state.get("passed_gates") or [])

    if state.get("plan") is None:
        _merge_update(state, _planner(state))
        gate = _gate_after("planner", policy, passed)
        if gate is not None:
            state["passed_gates"] = [*passed, gate]
            return state, gate

    if not state.get("planner_complete"):
        while not state["planner_complete"]:
            for question in list(state.get("pending_sub_questions") or []):
                _merge_update(
                    state,
                    _researcher(
                        {"sub_question": question, "otel_ctx": state.get("otel_ctx")}
                    ),
                )
            _merge_update(state, _planner_reflection(state))

    if not state.get("synthesis_complete"):
        while not state["synthesis_complete"]:
            _merge_update(state, _synthesizer(state))
            _merge_update(state, _synthesis_review(state))
    gate = _gate_after("synthesis_review", policy, passed)
    if gate is not None:
        state["passed_gates"] = [*passed, gate]
        return state, gate

    state["report"].latency_seconds = round(time.time() - started, 3)
    return state, None


def reset_for_redirect(state: ResearchState, question: str) -> None:
    """Re-arm a redirected run while preserving usage and frozen run settings."""
    state.update(
        {
            "question": question,
            "plan": None,
            "sub_results": [],
            "report": None,
            "passed_gates": [],
            "planner_iterations": 0,
            "pending_sub_questions": [],
            "planner_reflection": None,
            "planner_complete": False,
            "best_plan": None,
            "best_plan_score": None,
            "best_sub_results": [],
            "synthesis_iterations": 0,
            "synthesis_review": None,
            "synthesis_feedback": None,
            "synthesis_complete": False,
            "best_report": None,
            "best_report_score": None,
        }
    )


def serialize_state(state: ResearchState) -> dict:
    """Create a JSON-clean snapshot including every reflection candidate/counter."""
    out = dict(state)
    out.pop("otel_ctx", None)
    for key in (
        "plan",
        "report",
        "planner_reflection",
        "best_plan",
        "synthesis_review",
        "best_report",
    ):
        if isinstance(out.get(key), BaseModel):
            out[key] = json.loads(out[key].model_dump_json())
    for key in ("sub_results", "best_sub_results"):
        out[key] = [json.loads(item.model_dump_json()) for item in out.get(key) or []]
    out["usage_log"] = [asdict(usage) for usage in out.get("usage_log") or []]
    return out


def deserialize_state(snapshot: dict) -> ResearchState:
    """Validate and rebuild a snapshot; corrupt reflection state fails loudly."""
    try:
        enabled = _required_bool(snapshot, "reflection_enabled")
        planner_cap = _snapshot_non_negative_int(snapshot, "planner_reflection_max_iterations")
        synthesis_cap = _snapshot_non_negative_int(snapshot, "synthesis_review_max_iterations")
        planner_iterations = _snapshot_non_negative_int(snapshot, "planner_iterations")
        synthesis_iterations = _snapshot_non_negative_int(snapshot, "synthesis_iterations")
        if planner_iterations > planner_cap:
            raise ValueError("planner_iterations exceeds its cap")
        if synthesis_iterations > synthesis_cap:
            raise ValueError("synthesis_iterations exceeds its cap")

        state: ResearchState = {
            "question": _required_string(snapshot, "question"),
            "plan": _optional_model(snapshot.get("plan"), ResearchPlan),
            "sub_results": _model_list(snapshot.get("sub_results"), SubQuestionResult),
            "usage_log": [Usage(**usage) for usage in snapshot.get("usage_log") or []],
            "report": _optional_model(snapshot.get("report"), ResearchReport),
            "otel_ctx": None,
            "gate_policy": _gate_list(snapshot, "gate_policy"),
            "passed_gates": _gate_list(snapshot, "passed_gates"),
            "reflection_enabled": enabled,
            "planner_reflection_max_iterations": planner_cap,
            "synthesis_review_max_iterations": synthesis_cap,
            "planner_iterations": planner_iterations,
            "pending_sub_questions": _string_list(snapshot, "pending_sub_questions"),
            "planner_reflection": _optional_model(snapshot.get("planner_reflection"), PlannerReflection),
            "planner_complete": _required_bool(snapshot, "planner_complete"),
            "best_plan": _optional_model(snapshot.get("best_plan"), ResearchPlan),
            "best_plan_score": _optional_score(snapshot, "best_plan_score"),
            "best_sub_results": _model_list(snapshot.get("best_sub_results"), SubQuestionResult),
            "synthesis_iterations": synthesis_iterations,
            "synthesis_review": _optional_model(snapshot.get("synthesis_review"), SynthesisReview),
            "synthesis_feedback": _optional_string(snapshot, "synthesis_feedback"),
            "synthesis_complete": _required_bool(snapshot, "synthesis_complete"),
            "best_report": _optional_model(snapshot.get("best_report"), ResearchReport),
            "best_report_score": _optional_score(snapshot, "best_report_score"),
        }
        if "plan" in state["passed_gates"] and state["plan"] is None:
            raise ValueError("'plan' gate passed but plan is missing")
        if state["best_plan_score"] is not None and state["best_plan"] is None:
            raise ValueError("best_plan_score exists but best_plan is missing")
        if state["best_report_score"] is not None and state["best_report"] is None:
            raise ValueError("best_report_score exists but best_report is missing")
        if "synthesis" in state["passed_gates"] and (
            not state["synthesis_complete"] or state["report"] is None
        ):
            raise ValueError("'synthesis' gate passed before review completion")
        return state
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise ValueError(f"corrupt state snapshot: {exc}") from exc


def _gate_list(snapshot: dict, key: str) -> list[str]:
    value = snapshot.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(gate, str) for gate in value):
        raise ValueError(f"{key} must be a list of gate names, got {value!r}")
    unknown = [gate for gate in value if gate not in GATE_POINTS]
    if unknown:
        raise ValueError(f"{key} contains unknown gate names: {unknown!r}")
    return list(value)


def _merge_update(state: ResearchState, update: dict) -> None:
    """Fold one node update using the two declared LangGraph reducers."""
    for key, value in update.items():
        if key in {"sub_results", "usage_log"}:
            state[key] = [*(state.get(key) or []), *value]
        else:
            state[key] = value


def _gate_after(node_name: str, policy: set[str], passed: list[str]) -> str | None:
    for gate_name, node in GATE_POINTS.items():
        if node == node_name and gate_name in policy and gate_name not in passed:
            return gate_name
    return None


def _strip_fence(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _normalize_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def _copy_results(results: list[SubQuestionResult]) -> list[SubQuestionResult]:
    return [result.model_copy(deep=True) for result in results]


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _snapshot_non_negative_int(snapshot: dict, key: str) -> int:
    if key not in snapshot:
        raise ValueError(f"missing {key!r}")
    return _non_negative_int(snapshot[key], key)


def _required_bool(snapshot: dict, key: str) -> bool:
    value = snapshot.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _required_string(snapshot: dict, key: str) -> str:
    value = snapshot.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(snapshot: dict, key: str) -> str | None:
    value = snapshot.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _string_list(snapshot: dict, key: str) -> list[str]:
    value = snapshot.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return list(value)


def _optional_score(snapshot: dict, key: str) -> float | None:
    value = snapshot.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number or null")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return score


def _optional_model(value, model_type):
    if value is None:
        return None
    return model_type.model_validate(value)


def _model_list(value, model_type):
    if not isinstance(value, list):
        raise ValueError(f"{model_type.__name__} collection must be a list")
    return [model_type.model_validate(item) for item in value]
