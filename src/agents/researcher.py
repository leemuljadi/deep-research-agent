"""Researcher sub-agent: retrieves grounding content for one sub-question."""
from __future__ import annotations

from ..config import settings
from ..llm import Usage, chat_with_usage
from ..schemas import RetrieveToolInput, Source
from ..tools import execute

_SYNTHESIS_SYSTEM = (
    "You are a researcher. Given a sub-question and retrieved source passages, "
    "produce 1-3 concise, factual bullet findings. Only make claims supported by "
    "the retrieved passages; mark unsupported claims as 'UNSUPPORTED'. "
    "Return ONLY valid JSON matching: "
    '{"sub_question": str, "findings": [str, ...], "sources": '
    '[{"title": str, "url": str|null, "snippet": str, "score": float|null}, ...]}'
)

def research_sub_question(sub_question: str) -> tuple[list[str], list[Source], Usage]:
    """Retrieve grounding chunks and return (findings, sources, usage)."""
    result = execute(RetrieveToolInput(query=sub_question, top_k=settings.top_k))
    hits = result.payload if result.state == "ok" else []
    if not hits:
        return [f"No grounding found for: {sub_question}"], [], Usage()

    passages = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
    prompt = (
        f"SUB-QUESTION: {sub_question}\n\n"
        f"RETRIEVED PASSAGES:\n{passages}\n\n"
        "Produce findings grounded only in these passages, with source metadata."
    )
    raw, usage = chat_with_usage(
        [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    import json

    try:
        data = json.loads(_strip_fence(raw))
        if not isinstance(data, dict):
            raise TypeError(f"expected JSON object, got {type(data).__name__}")
    except (json.JSONDecodeError, TypeError):
        # Graceful fallback: surface raw retrieval so the pipeline still works.
        return [f"(parse-fallback) {sub_question}: {raw[:200]}"], [
            Source(title=h.title or h.doc_id, url=h.url, snippet=h.text[:200], score=h.score)
            for h in hits[:3]
        ], usage

    sources = [
        Source(
            title=s.get("title") or h.doc_id,
            url=s.get("url"),
            snippet=s.get("snippet") or h.text[:200],
            score=s.get("score"),
        )
        for s, h in zip(data.get("sources", []), hits)
    ]
    return data.get("findings", []), sources, usage


def _strip_fence(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()