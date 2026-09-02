"""Synthesizer: lead agent writes the final grounded, structured report."""
from __future__ import annotations

import json

from ..llm import Usage, chat_with_usage
from ..schemas import ResearchPlan, ResearchReport, Source

_SYNTH_SYSTEM = (
    "You are a senior analyst. Given a research objective and the findings from "
    "multiple parallel researchers, write a structured report. Ground every "
    "finding in the provided sources with citations [n]. Do not invent facts. "
    "Return ONLY valid JSON matching: "
    '{"summary": str, "findings": [str, ...], '
    '"sources": [{"title": str, "url": str|null, "snippet": str, "score": float|null}], '
    '"confidence": float}'
)


def synthesize(
    plan: ResearchPlan,
    sub_results: list[tuple[str, list[str], list[Source]]],
) -> tuple[ResearchReport, Usage]:
    blocks = []
    for sub_q, findings, _srcs in sub_results:
        blocks.append(f"## {sub_q}\n" + "\n".join(f"- {f}" for f in findings))
    joined = "\n\n".join(blocks)

    raw, usage = chat_with_usage(
        [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"OBJECTIVE: {plan.objective}\n\nRESEARCHER FINDINGS:\n{joined}"
                ),
            },
        ],
        temperature=0.2,
    )
    data = json.loads(_strip_fence(raw))
    sources = [Source(**s) for s in data.get("sources", [])]
    return (
        ResearchReport(
            summary=data.get("summary", ""),
            findings=data.get("findings", []),
            sources=sources,
            confidence=float(data.get("confidence", 0.5)),
            cost_usd=0.0,
            latency_seconds=0.0,
        ),
        usage,
    )


def _strip_fence(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()