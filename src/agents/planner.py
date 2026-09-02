"""Lead agent: decomposes a research question into a structured plan."""
from __future__ import annotations

from ..llm import Usage, chat_with_usage
from ..schemas import ResearchPlan

_PLANNER_SYSTEM = (
    "You are a meticulous research planner. Decompose the user's question into "
    "independent, answerable sub-questions. Restate the objective precisely. "
    "Return ONLY valid JSON matching: "
    '{"objective": str, "sub_questions": [str, ...]}.'
)


def plan(question: str) -> tuple[ResearchPlan, Usage]:
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": question},
    ]
    raw, usage = chat_with_usage(messages, temperature=0.0)
    # Tolerate markdown-fenced JSON.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return ResearchPlan.model_validate_json(cleaned), usage