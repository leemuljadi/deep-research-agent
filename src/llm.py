"""LiteLLM routing for chat and embeddings across model providers.

An optional ordered chat deployment provides primary-to-fallback failover.
``chat_with_usage`` returns provider token counts and cost for tracing and
evaluation, and every capture checks the run-scoped cost-cap accumulator
(AD-15) — the single enforcement home shared with tool-cost capture.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import litellm
from litellm import Router

from .config import settings

# Quiet LiteLLM's verbose debug logging.
litellm.suppress_debug_info = True
litellm.drop_params = True


@dataclass(frozen=True)
class Usage:
    """Token usage and cost for one completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class CostCapExceeded(RuntimeError):
    """A run crossed its cost cap (AD-15) — run-terminal, never degraded."""


# Run-scoped cap accumulator (AD-15): one registry, one shared run-total per
# run id, plus a run token marking the job currently executing in this
# process. Same singleton class of module state as `get_router()`/tracer —
# set by the worker before execution, cleared in `finally`.
_cap_lock = threading.Lock()
_run_scope: dict[str, dict[str, float | None]] = {}
_current_run: str | None = None


def current_run_id() -> str | None:
    """The run token capture paths attribute usage to (None = no scope)."""
    return _current_run


def set_run_cap(run_id: str, cap_usd: float | None) -> None:
    """Set the run token and its cap; None = uncapped. Resets the
    accumulator — the worker calls it before executing a job so each claim
    starts at zero."""
    global _current_run
    with _cap_lock:
        _run_scope[run_id] = {"cap": cap_usd, "spent": 0.0}
        _current_run = run_id


def clear_run_cap(run_id: str) -> None:
    """Drop a run's scope and token entirely (worker `finally`)."""
    global _current_run
    with _cap_lock:
        _run_scope.pop(run_id, None)
        if _current_run == run_id:
            _current_run = None


def get_run_cap() -> float | None:
    """The current run's cap, or None when uncapped / no scope exists."""
    with _cap_lock:
        scope = _run_scope.get(_current_run) if _current_run else None
    return None if scope is None else scope["cap"]


def seed_run_spend(cost_usd: float) -> None:
    """Re-seed the current run's accumulator from a snapshot's `usage_log`
    sum — a run resumed after a gate pause must keep checking against the
    same total."""
    with _cap_lock:
        scope = _run_scope.get(_current_run) if _current_run else None
        if scope is not None:
            scope["spent"] = float(cost_usd)


def check_cost(run_id: str | None, cost_usd: float) -> None:
    """Fold one captured cost into the shared run-total and trip the cap.

    The single check both capture paths call (llm.py Usage, tools.py story 6).
    Trips on `total >= cap` — exactly reaching the cap trips, no free overage.
    Zero-cost captures never trip (a zero-cost run's total stays 0.0, and a
    cap of 0 only trips on a cost-bearing call). Raises before returning so
    the caller never sees a capped result; overshoot is bounded by one
    in-flight cost-bearing call.
    """
    with _cap_lock:
        scope = _run_scope.get(run_id) if run_id else None
        if scope is None:
            return  # no scope registered: uncapped (default path)
        scope["spent"] += float(cost_usd)
        cap = scope["cap"]
        total = scope["spent"]
    if cap is not None and total > 0 and total >= cap:
        raise CostCapExceeded(
            f"cost cap exceeded for run {run_id}: "
            f"accumulated ${total:.4f} >= cap ${cap:.4f}"
        )


def register_usage(run_id: str | None, usage: Usage) -> None:
    """Capture-path entry point: register one Usage then check the cap."""
    check_cost(run_id, usage.cost_usd)


def _build_router() -> Router:
    """Build a LiteLLM Router with optional fallback across providers."""
    models: list[dict[str, Any]] = [
        {
            "model_name": "chat",
            "litellm_params": {"model": settings.chat_model, "order": 1},
        },
        {
            "model_name": "embed",
            "litellm_params": {"model": settings.embed_model},
        },
    ]
    # Fallback: route to a second provider when the primary is unavailable.
    if settings.chat_model_fallback:
        models.append(
            {
                "model_name": "chat",
                "litellm_params": {
                    "model": settings.chat_model_fallback,
                    "order": 2,
                },
            }
        )
    return Router(model_list=models)


_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = _build_router()
    return _router


def chat_with_usage(
    messages: list[dict[str, str]], *, temperature: float = 0.3
) -> tuple[str, Usage]:
    """Send a chat completion through the router.

    Returns (assistant text, usage) — usage carries real token counts and the
    provider-priced cost LiteLLM attaches to the response.
    """
    resp = get_router().completion(
        model="chat",
        messages=messages,
        temperature=temperature,
    )
    u = getattr(resp, "usage", None)
    usage = Usage(
        prompt_tokens=(getattr(u, "prompt_tokens", 0) or 0),
        completion_tokens=(getattr(u, "completion_tokens", 0) or 0),
        total_tokens=(getattr(u, "total_tokens", 0) or 0),
        cost_usd=(getattr(resp, "_hidden_params", None) or {}).get("response_cost") or 0.0,
    )
    register_usage(_current_run, usage)
    return resp.choices[0].message.content or "", usage


def chat(messages: list[dict[str, str]], *, temperature: float = 0.3) -> str:
    """Send a chat completion through the router. Returns the assistant text."""
    return chat_with_usage(messages, temperature=temperature)[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings via the configured embedding model."""
    resp = get_router().embedding(model="embed", input=texts)
    # LiteLLM returns data as a list of {embedding: [...]}
    embeddings = [item["embedding"] for item in resp["data"]]
    cost = (getattr(resp, "_hidden_params", None) or {}).get("response_cost") or 0.0
    register_usage(_current_run, Usage(total_tokens=len(texts), cost_usd=cost))
    return embeddings


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]