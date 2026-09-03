"""One tools-executor boundary (AD-16).

Agents call tools ONLY through `execute()` — the single choke point that owns
per-call timeout, a terminal ToolResult for every outcome (ok / not_found /
disabled / error / timeout — never an open-ended await), cost capture into the
run's AD-15 cap accumulator via `llm.check_cost`, and span parenting (AD-8).

No agent imports `search.py`, `db.py`, or any HTTP client / vendor SDK
directly; the executor is their only sanctioned consumer.
"""
from __future__ import annotations

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable

from .config import settings
from .llm import check_cost, current_run_id
from .schemas import (
    ReadDocumentToolInput,
    RetrieveToolInput,
    ToolResult,
    WebSearchToolInput,
)
from .search import SearchResult, get_document_text, hybrid_search
from .tracing import trace

ToolInput = RetrieveToolInput | ReadDocumentToolInput | WebSearchToolInput

_STATE_OK = "ok"
_STATE_TIMEOUT = "timeout"
_STATE_ERROR = "error"
_STATE_NOT_FOUND = "not_found"
_STATE_DISABLED = "disabled"


def _exec_retrieve(tool_input: RetrieveToolInput) -> list[SearchResult]:
    """The `retrieve` tool: hybrid search over the indexed corpus (AD-4)."""
    return hybrid_search(tool_input.query, top_k=tool_input.top_k)


def _exec_read_document(tool_input: ReadDocumentToolInput) -> str | None:
    """The `read_document` tool: full text of one stored document (AD-4)."""
    return get_document_text(tool_input.doc_id)


def _exec_web_search(tool_input: WebSearchToolInput) -> ToolResult:
    """The `web_search` tool: provider-backed, additive (AD-13).

    No provider wiring exists yet (deliberately out of this story); the
    free-local default path returns a terminal `disabled` result so the
    zero-key configuration stays clean and callers degrade gracefully.
    """
    return ToolResult(
        tool="web_search", state=_STATE_DISABLED,
        detail="no web-search provider configured (free-local default, AD-13)",
    )


_EXECUTORS: dict[str, Callable[[ToolInput], object]] = {
    "retrieve": _exec_retrieve,
    "read_document": _exec_read_document,
    "web_search": _exec_web_search,
}

_TOOL_NAMES: dict[type, str] = {
    RetrieveToolInput: "retrieve",
    ReadDocumentToolInput: "read_document",
    WebSearchToolInput: "web_search",
}

# Local tools are zero-cost (local DB reads); the capture hook runs for every
# tool so the first priced provider (story 6-follow-up) only adds its entry.
_TOOL_COST_USD: dict[str, float] = {
    "retrieve": 0.0,
    "read_document": 0.0,
    "web_search": 0.0,
}


def _tool_cost(name: str) -> float:
    """Per-call cost of a tool: 0.0 for local tools, provider cost later."""
    return _TOOL_COST_USD.get(name, 0.0)


def execute(tool_input: ToolInput, *, run_id: str | None = None) -> ToolResult:
    """Run one tool call through the AD-16 choke point.

    Guarantees, for every call:
      - a terminal ToolResult (ok / timeout / error / not_found / disabled) —
        a provider that hangs cannot reach the run;
      - the call wrapped in a `trace("tool.<name>")` span; run_id and the
        run's OTel context attach the span to the run when available (AD-8);
      - the tool's cost folded into the shared run-total via `llm.check_cost`
        on EVERY terminal outcome — timeout, error, and success alike — so a
        provider that metered a request that then timed out is still charged
        (AD-15 overshoot bound);
      - unknown tool names are terminal `error` results, not crashes.

    run_id defaults to the worker/CLI run token (`llm.current_run_id()`);
    callers may override. With no run scope, check_cost is a no-op (uncapped).
    """
    name = _TOOL_NAMES.get(type(tool_input), "unknown")
    executor = _EXECUTORS.get(name)
    if executor is None:
        return ToolResult(
            tool="unknown", state=_STATE_ERROR,
            detail=f"unknown tool: {name}",
        )
    if run_id is None:
        run_id = current_run_id()

    attrs: dict[str, str] = {"run_id": run_id} if run_id else {}
    cost = _tool_cost(name)
    t0 = time.monotonic()
    result: ToolResult | None = None
    try:
        with trace(f"tool.{name}", **attrs):
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                try:
                    future = pool.submit(executor, tool_input)
                except Exception as exc:  # e.g. can't start new thread
                    result = ToolResult(
                        tool=name, state=_STATE_ERROR, detail=str(exc),
                    )
                else:
                    try:
                        raw = future.result(timeout=settings.tool_timeout_seconds)
                    except futures.TimeoutError:
                        result = ToolResult(
                            tool=name, state=_STATE_TIMEOUT,
                            detail=f"tool timed out after {settings.tool_timeout_seconds}s",
                        )
                    except Exception as exc:  # noqa: BLE001 — terminal
                        # A builtin TimeoutError from inside the tool body
                        # (socket timeout) is a tool error, not the executor's
                        # window — distinguish by elapsed time.
                        elapsed = time.monotonic() - t0
                        state = (
                            _STATE_TIMEOUT
                            if elapsed >= settings.tool_timeout_seconds
                            and isinstance(exc, TimeoutError)
                            else _STATE_ERROR
                        )
                        result = ToolResult(
                            tool=name, state=state, detail=str(exc),
                        )
                    else:
                        if isinstance(raw, ToolResult):
                            result = raw
                        elif raw is None:
                            result = ToolResult(tool=name, state=_STATE_NOT_FOUND)
                        else:
                            result = ToolResult(tool=name, state=_STATE_OK, payload=raw)
            finally:
                # Do NOT wait on exit: a hung tool must never block the run
                # past the timeout (the AD-16 terminal-state guarantee).
                pool.shutdown(wait=False, cancel_futures=True)

        if result is None:
            result = ToolResult(tool=name, state=_STATE_NOT_FOUND)
    except Exception as exc:  # belt-and-braces: never escape the choke point
        result = ToolResult(tool=name, state=_STATE_ERROR, detail=str(exc))
    check_cost(run_id, cost)
    return result