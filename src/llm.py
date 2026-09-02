"""LiteLLM routing for chat and embeddings across model providers.

An optional ordered chat deployment provides primary-to-fallback failover.
``chat_with_usage`` returns provider token counts and cost for tracing and
evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def _build_router() -> Router:
    """Build a LiteLLM Router with optional fallback across providers."""
    models: list[dict[str, Any]] = [
        {
            "model_name": "chat",
            "litellm_params": {"model": settings.chat_model},
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
                "litellm_params": {"model": settings.chat_model_fallback},
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
    return resp.choices[0].message.content or "", usage


def chat(messages: list[dict[str, str]], *, temperature: float = 0.3) -> str:
    """Send a chat completion through the router. Returns the assistant text."""
    return chat_with_usage(messages, temperature=temperature)[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings via the configured embedding model."""
    resp = get_router().embedding(model="embed", input=texts)
    # LiteLLM returns data as a list of {embedding: [...]}
    return [item["embedding"] for item in resp["data"]]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]