"""Executor matrix for the tools boundary (story 6, AD-16).

Covers: each tool × each terminal state, the timeout path, cost capture
through `llm.check_cost`, span parenting, the researcher's degradation
mapping, and the import-scan guard (no agent imports search/db/HTTP).
"""
from __future__ import annotations

import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src import tools
from src.agents import researcher
from src.config import Settings
from src.schemas import (
    ReadDocumentToolInput,
    RetrieveToolInput,
    ToolResult,
    WebSearchToolInput,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"


def _ok(payload) -> ToolResult:
    return ToolResult(tool="retrieve", state="ok", payload=payload)


class ExecutorMatrixTests(unittest.TestCase):
    """The execute() choke point: dispatch, terminal states, wrapping."""

    def test_retrieve_happy_path_returns_ok_with_payload(self) -> None:
        hits = [object()]
        with (
            patch.object(tools, "hybrid_search", return_value=hits) as hs,
            patch.object(tools, "trace", return_value=nullcontext(object())),
            patch.object(tools, "check_cost") as cost,
        ):
            result = tools.execute(RetrieveToolInput(query="q", top_k=3))
        self.assertEqual(result.tool, "retrieve")
        self.assertEqual(result.state, "ok")
        self.assertIs(result.payload, hits)
        hs.assert_called_once_with("q", top_k=3)
        cost.assert_called_once_with(None, 0.0)

    def test_read_document_hit_returns_ok_with_text(self) -> None:
        with (
            patch.object(tools, "get_document_text", return_value="doc text") as gt,
            patch.object(tools, "trace", return_value=nullcontext(object())),
            patch.object(tools, "check_cost"),
        ):
            result = tools.execute(ReadDocumentToolInput(doc_id="doc-1"))
        self.assertEqual(result.state, "ok")
        self.assertEqual(result.payload, "doc text")
        gt.assert_called_once_with("doc-1")

    def test_read_document_miss_is_terminal_not_found(self) -> None:
        with (
            patch.object(tools, "get_document_text", return_value=None),
            patch.object(tools, "trace", return_value=nullcontext(object())),
            patch.object(tools, "check_cost"),
        ):
            result = tools.execute(ReadDocumentToolInput(doc_id="missing"))
        self.assertEqual(result.tool, "read_document")
        self.assertEqual(result.state, "not_found")
        self.assertIsNone(result.payload)

    def test_web_search_no_provider_is_terminal_disabled(self) -> None:
        """Zero-key default: no paid call, no error — the AD-13 deliverable."""
        with (
            patch.object(tools, "trace", return_value=nullcontext(object())),
            patch.object(tools, "check_cost"),
        ):
            result = tools.execute(WebSearchToolInput(query="anything"))
        self.assertEqual(result.tool, "web_search")
        self.assertEqual(result.state, "disabled")
        self.assertIsNone(result.payload)

    def test_web_search_provider_error_is_terminal_error(self) -> None:
        def boom(tool_input):
            raise RuntimeError("provider exploded")

        saved = tools._EXECUTORS["web_search"]
        tools._EXECUTORS["web_search"] = boom
        try:
            with (
                patch.object(tools, "trace", return_value=nullcontext(object())),
                patch.object(tools, "check_cost"),
            ):
                result = tools.execute(WebSearchToolInput(query="q"))
        finally:
            tools._EXECUTORS["web_search"] = saved
        self.assertEqual(result.state, "error")
        self.assertIn("provider exploded", result.detail)

    def test_unknown_tool_name_is_error_result_not_crash(self) -> None:
        class BogusInput:
            pass

        with patch.object(tools, "trace", return_value=nullcontext(object())):
            result = tools.execute(BogusInput())  # type: ignore[arg-type]
        self.assertEqual(result.state, "error")
        self.assertIn("unknown", result.detail or "")

    def test_hanging_tool_terminates_as_timeout(self) -> None:
        """The AD-16 terminal-state guarantee: a hung tool becomes a
        terminal timeout result inside the timeout window — never a hang."""
        import threading

        release = threading.Event()

        def hang(tool_input):
            # Long past the executor's 0.2s window: proves the timeout path
            # fires without the tool finishing. Event-set at the end so the
            # worker thread exits promptly and doesn't stall test teardown.
            release.wait(timeout=10)

        saved = tools._EXECUTORS["retrieve"]
        tools._EXECUTORS["retrieve"] = hang
        try:
            mutable = Settings()
            object.__setattr__(mutable, "tool_timeout_seconds", 0.2)
            with (
                patch.object(tools, "trace", return_value=nullcontext(object())),
                patch.object(tools, "check_cost"),
                patch.object(tools, "settings", mutable),
            ):
                result = tools.execute(RetrieveToolInput(query="q"))
        finally:
            tools._EXECUTORS["retrieve"] = saved
        self.assertEqual(result.state, "timeout")
        self.assertIn("0.2", result.detail or "")
        release.set()  # free the sleeper thread so the suite exits promptly

    def test_tool_cost_captured_through_llm_check_cost(self) -> None:
        """Every completed call folds its cost through the shared accumulator."""
        with (
            patch.object(tools, "hybrid_search", return_value=[]),
            patch.object(tools, "trace", return_value=nullcontext(object())),
            patch.object(tools, "check_cost") as cost,
        ):
            tools.execute(RetrieveToolInput(query="q"), run_id="run-1")
        cost.assert_called_once_with("run-1", 0.0)

    def test_cost_cap_exceeded_propagates_from_check_cost(self) -> None:
        from src.llm import CostCapExceeded

        with (
            patch.object(tools, "hybrid_search", return_value=[]),
            patch.object(tools, "trace", return_value=nullcontext(object())),
            patch.object(tools, "check_cost", side_effect=CostCapExceeded("cap")),
        ):
            with self.assertRaises(CostCapExceeded):
                tools.execute(RetrieveToolInput(query="q"), run_id="run-1")

    def test_span_parented_via_trace_with_run_context(self) -> None:
        """Each call wraps in trace("tool.<name>") with the run attribute set."""
        with (
            patch.object(tools, "hybrid_search", return_value=[]),
            patch.object(tools, "trace", return_value=nullcontext(object())) as tr,
            patch.object(tools, "check_cost"),
        ):
            tools.execute(RetrieveToolInput(query="q"), run_id="run-9")
        tr.assert_called_once_with("tool.retrieve", run_id="run-9")

    def test_web_search_disabled_result_reused_not_rewrapped(self) -> None:
        """An executor returning a ToolResult (the disabled stub) surfaces as-is."""
        disabled = ToolResult(tool="web_search", state="disabled", detail="d")
        saved = tools._EXECUTORS["web_search"]
        tools._EXECUTORS["web_search"] = lambda ti: disabled
        try:
            with (
                patch.object(tools, "trace", return_value=nullcontext(object())),
                patch.object(tools, "check_cost"),
            ):
                result = tools.execute(WebSearchToolInput(query="q"))
        finally:
            tools._EXECUTORS["web_search"] = saved
        self.assertIs(result, disabled)


class ToolResultShapeTests(unittest.TestCase):
    def test_tool_result_rejects_unknown_state(self) -> None:
        with self.assertRaises(ValidationError):
            ToolResult(tool="retrieve", state="exploded")

    def test_tool_result_defaults(self) -> None:
        result = ToolResult(tool="web_search", state="disabled")
        self.assertIsNone(result.payload)
        self.assertIsNone(result.detail)


class ConfigTimeoutTests(unittest.TestCase):
    """TOOL_TIMEOUT_SECONDS guarded parse (mirrors _cost_cap_env, AD-9)."""

    def _parse(self, raw: str | None) -> float:
        from src.config import _tool_timeout_env

        return _tool_timeout_env(raw)

    def test_unset_and_empty_default_to_30(self) -> None:
        self.assertEqual(self._parse(None), 30.0)
        self.assertEqual(self._parse("  "), 30.0)

    def test_garbage_defaults_to_30(self) -> None:
        self.assertEqual(self._parse("abc"), 30.0)

    def test_non_finite_defaults_to_30(self) -> None:
        self.assertEqual(self._parse("nan"), 30.0)
        self.assertEqual(self._parse("inf"), 30.0)
        self.assertEqual(self._parse("-inf"), 30.0)

    def test_non_positive_defaults_to_30(self) -> None:
        self.assertEqual(self._parse("0"), 30.0)
        self.assertEqual(self._parse("-5"), 30.0)

    def test_valid_value_parses(self) -> None:
        self.assertEqual(self._parse("12.5"), 12.5)


class ResearcherIntegrationTests(unittest.TestCase):
    """Tool failure states map to the researcher's existing empty-hits path."""

    def _run_with_state(self, state: str, payload=None):
        result = ToolResult(tool="retrieve", state=state, payload=payload)
        with (
            patch.object(researcher, "execute", return_value=result),
            patch.object(researcher, "chat_with_usage", return_value=("", researcher.Usage())),
        ):
            return researcher.research_sub_question("q?")

    def test_ok_payload_flows_into_research(self) -> None:
        class Hit:
            doc_id, title, url, text, score = "d1", "T", None, "body", 1.0

        findings, sources, _usage = self._run_with_state(
            "ok",
            payload=[Hit()],
        )
        self.assertEqual(findings, ["(parse-fallback) q?: "])
        self.assertEqual(sources[0].title, "T")

    def test_failure_states_degrade_to_no_grounding(self) -> None:
        for state in ("disabled", "error", "timeout", "not_found"):
            with self.subTest(state=state):
                findings, sources, usage = self._run_with_state(state)
                self.assertEqual(
                    findings, [f"No grounding found for: q?"]
                )
                self.assertEqual(sources, [])
                self.assertEqual(usage.total_tokens, 0)

    def test_researcher_calls_only_through_executor(self) -> None:
        """The direct hybrid_search call is gone from the researcher."""
        with patch.object(researcher, "execute") as ex:
            ex.return_value = ToolResult(tool="retrieve", state="ok", payload=[])
            with patch.object(researcher, "chat_with_usage", return_value=("", researcher.Usage())):
                researcher.research_sub_question("q?")
            ex.assert_called_once()


class ImportBoundaryTests(unittest.TestCase):
    """No agent imports search/db/HTTP clients — AD-16, executor-only access."""

    _FORBIDDEN = ("search", "db", "httpx", "requests", "urllib", "aiohttp")

    def test_agent_modules_import_no_storage_or_http(self) -> None:
        agents_dir = _SRC_DIR / "agents"
        for path in sorted(agents_dir.glob("*.py")):
            text = path.read_text()
            for line in text.splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                for mod in self._FORBIDDEN:
                    if f".{mod}" in line or f" {mod} " in line or line.endswith(
                        f" {mod}"
                    ) or f"import {mod}" in line:
                        # relative import of the shared module names is only
                        # sanctioned in tools.py
                        self.fail(
                            f"{path.name}: forbidden import '{mod}': {stripped}"
                        )

    def test_tools_py_owns_the_search_import(self) -> None:
        text = (_SRC_DIR / "tools.py").read_text()
        self.assertIn("from .search import", text)
        self.assertIn("hybrid_search", text)
        self.assertIn("get_document_text", text)


if __name__ == "__main__":
    unittest.main()