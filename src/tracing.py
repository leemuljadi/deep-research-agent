"""Observability: OpenTelemetry spans exported to Langfuse (OTLP).

Every agent step opens a span carrying latency, token usage and cost
(`record_cost`). Spans are emitted through the OpenTelemetry API; when Langfuse
keys are configured, the SDK is wired with an OTLP exporter pointed at
Langfuse's ingestion endpoint, which renders the full run tree (plan ->
parallel research -> synthesise). Without keys, the API's default no-op
provider makes tracing free — the local stack runs uninstrumented.
"""
from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace

from .config import settings

# In-memory mirror of recent spans (tests / quick inspection).
_traces: list[dict] = []


@dataclass
class SpanHandle:
    """Wraps the OTel span plus a local record dict."""

    otel: otel_trace.Span
    record: dict = field(default_factory=dict)


_tracer: otel_trace.Tracer | None = None
_provider = None  # reference so the SDK exporter is not GC'd


def _build_langfuse_provider():
    """Configure the OTel SDK with a Langfuse OTLP exporter, if keys exist."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return None  # exporter package not installed; stay uninstrumented

    host = (settings.langfuse_host or "https://cloud.langfuse.com").rstrip("/")
    auth = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "deep-research-agent"})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{host}/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {auth}"},
            )
        )
    )
    otel_trace.set_tracer_provider(provider)
    return provider


def _ensure_tracer() -> otel_trace.Tracer:
    global _tracer, _provider
    if _tracer is None:
        if _provider is None:
            _provider = _build_langfuse_provider()
        _tracer = otel_trace.get_tracer("deep-research-agent")
    return _tracer


def current_context() -> object:
    """Snapshot the current OTel context (parent-child spans across threads)."""
    return otel_context.get_current()


@contextmanager
def trace(
    name: str, *, ctx: object | None = None, **attrs: str
) -> Iterator[SpanHandle]:
    """Record a span for one agent step.

    Backs onto OpenTelemetry (exported to Langfuse when configured); the API's
    no-op provider keeps this free otherwise. Also mirrors into an in-memory
    list for `get_traces()`.
    """
    tracer = _ensure_tracer()
    token = otel_context.attach(ctx) if ctx is not None else None
    try:
        # start_span returns the span object itself (start_as_current_span
        # returns a context manager, not a Span).
        otel_span = tracer.start_span(name)
        handle = SpanHandle(
            otel=otel_span, record={"name": name, **attrs, "start": time.time()}
        )
        try:
            for k, v in attrs.items():
                otel_span.set_attribute(k, v)
            yield handle
        finally:
            latency = time.time() - handle.record["start"]
            handle.record["latency_s"] = latency
            otel_span.set_attribute("latency_s", latency)
            otel_span.end()
            _traces.append(handle.record)
    finally:
        if token is not None:
            otel_context.detach(token)


def record_cost(span: SpanHandle | None, *, tokens: int, cost_usd: float) -> None:
    """Attach token usage and cost to a span (OTel attributes + local record)."""
    if span is None:
        return
    span.record["tokens"] = tokens
    span.record["cost_usd"] = cost_usd
    span.otel.set_attribute("gen_ai.usage.total_tokens", tokens)
    span.otel.set_attribute("gen_ai.usage.cost", cost_usd)


def get_traces() -> list[dict]:
    return list(_traces)