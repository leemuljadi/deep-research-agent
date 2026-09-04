from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.exceptions import ToolError

from src.schemas import RunStatus, RunStatusResponse, RunTransitionResponse


UUID = "d0a1b2c3-d4e5-4678-9abc-def012345678"
SECRET = "story-eleven-test-secret-at-least-32-bytes"


def _job_row(**overrides) -> dict:
    row = {
        "id": UUID,
        "question": "What is RAG?",
        "status": "waiting_for_input",
        "result": None,
        "error": None,
        "created_at": datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 9, 4, 12, 5, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _jwt(secret: str, *, expires_at: int, subject: str = "test-client") -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(
        json.dumps({"sub": subject, "exp": expires_at}, separators=(",", ":")).encode()
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = _b64url(
        hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{signature}"


class McpToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        import src.mcp_server as mcp_server

        self.module = mcp_server
        self.server = mcp_server.build_mcp_server(None)

    async def test_inventory_is_exactly_the_five_job_boundary_verbs(self) -> None:
        async with Client(self.server) as client:
            tools = await client.list_tools()
        self.assertEqual(
            {tool.name for tool in tools},
            {"submit", "poll", "approve", "redirect", "cancel"},
        )

    async def test_submit_enqueues_without_executing_graph(self) -> None:
        with patch.object(self.module.db, "enqueue_job", return_value=UUID) as enqueue:
            async with Client(self.server) as client:
                result = await client.call_tool(
                    "submit", {"question": "  Why does RAG work?  "}
                )
        self.assertEqual(result.data.run_id, UUID)
        enqueue.assert_called_once_with("Why does RAG work?")

    async def test_submit_rejects_blank_before_enqueue(self) -> None:
        with patch.object(self.module.db, "enqueue_job") as enqueue:
            async with Client(self.server) as client:
                with self.assertRaises(ToolError):
                    await client.call_tool("submit", {"question": "   "})
        enqueue.assert_not_called()

    async def test_poll_returns_the_shared_status_shape(self) -> None:
        with patch.object(
            self.module.db, "get_job", return_value=_job_row()
        ) as get_job:
            async with Client(self.server) as client:
                result = await client.call_tool("poll", {"run_id": UUID})
        response = RunStatusResponse.model_validate(result.structured_content)
        self.assertEqual(response.status, RunStatus.WAITING_FOR_INPUT)
        get_job.assert_called_once_with(UUID)

    async def test_poll_unknown_run_is_a_tool_error(self) -> None:
        with patch.object(self.module.db, "get_job", return_value=None):
            async with Client(self.server) as client:
                with self.assertRaisesRegex(ToolError, "run not found"):
                    await client.call_tool("poll", {"run_id": UUID})

    async def test_approve_uses_existing_decision_and_resume_transitions(self) -> None:
        with (
            patch.object(self.module.db, "get_job", return_value=_job_row()),
            patch.object(
                self.module.db, "record_decision", return_value=True
            ) as record,
            patch.object(self.module.db, "resume_run", return_value=True) as resume,
        ):
            async with Client(self.server) as client:
                result = await client.call_tool("approve", {"run_id": UUID})
        response = RunTransitionResponse.model_validate(result.structured_content)
        self.assertEqual(response.status, RunStatus.QUEUED)
        record.assert_called_once_with(UUID, "approve", {})
        resume.assert_called_once_with(UUID)

    async def test_approve_conflict_does_not_resume(self) -> None:
        with (
            patch.object(self.module.db, "get_job", return_value=_job_row()),
            patch.object(self.module.db, "record_decision", return_value=False),
            patch.object(self.module.db, "resume_run") as resume,
        ):
            async with Client(self.server) as client:
                with self.assertRaisesRegex(ToolError, "not waiting_for_input"):
                    await client.call_tool("approve", {"run_id": UUID})
        resume.assert_not_called()

    async def test_redirect_normalizes_question_and_reuses_transitions(self) -> None:
        with (
            patch.object(self.module.db, "get_job", return_value=_job_row()),
            patch.object(
                self.module.db, "record_decision", return_value=True
            ) as record,
            patch.object(self.module.db, "resume_run", return_value=True) as resume,
        ):
            async with Client(self.server) as client:
                result = await client.call_tool(
                    "redirect", {"run_id": UUID, "question": "  New question  "}
                )
        self.assertEqual(result.data.question, "New question")
        record.assert_called_once_with(
            UUID, "redirect", {"question": "New question"}
        )
        resume.assert_called_once_with(UUID, resume_question="New question")

    async def test_redirect_rejects_blank_before_writes(self) -> None:
        with (
            patch.object(self.module.db, "get_job") as get_job,
            patch.object(self.module.db, "record_decision") as record,
            patch.object(self.module.db, "resume_run") as resume,
        ):
            async with Client(self.server) as client:
                with self.assertRaises(ToolError):
                    await client.call_tool(
                        "redirect", {"run_id": UUID, "question": "  "}
                    )
        get_job.assert_not_called()
        record.assert_not_called()
        resume.assert_not_called()

    async def test_cancel_reuses_existing_conditional_transition(self) -> None:
        with (
            patch.object(self.module.db, "get_job", return_value=_job_row()),
            patch.object(self.module.db, "cancel_run", return_value=True) as cancel_run,
        ):
            async with Client(self.server) as client:
                result = await client.call_tool("cancel", {"run_id": UUID})
        self.assertEqual(result.data.status, RunStatus.CANCELLED)
        cancel_run.assert_called_once_with(UUID)

    async def test_cancel_running_run_is_a_conflict(self) -> None:
        with (
            patch.object(
                self.module.db,
                "get_job",
                return_value=_job_row(status="running"),
            ),
            patch.object(self.module.db, "cancel_run", return_value=False),
        ):
            async with Client(self.server) as client:
                with self.assertRaisesRegex(ToolError, "not cancellable"):
                    await client.call_tool("cancel", {"run_id": UUID})


class McpAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_hs256_verifier_accepts_valid_token(self) -> None:
        from src.mcp_server import _jwt_verifier

        verifier = _jwt_verifier(SECRET)
        token = _jwt(SECRET, expires_at=int(time.time()) + 60)
        self.assertIsNotNone(await verifier.verify_token(token))

    async def test_hs256_verifier_rejects_bad_and_expired_tokens(self) -> None:
        from src.mcp_server import _jwt_verifier

        verifier = _jwt_verifier(SECRET)
        bad_signature = _jwt(
            "a-different-secret-at-least-32-bytes",
            expires_at=int(time.time()) + 60,
        )
        expired = _jwt(SECRET, expires_at=int(time.time()) - 60)
        self.assertIsNone(await verifier.verify_token("not-a-jwt"))
        self.assertIsNone(await verifier.verify_token(bad_signature))
        self.assertIsNone(await verifier.verify_token(expired))


class McpMountTests(unittest.TestCase):
    def test_mcp_mount_coexists_with_run_routes(self) -> None:
        import server

        paths = {getattr(route, "path", None) for route in server.app.routes}
        self.assertIn("/mcp", paths)
        self.assertIn("/runs/{run_id}", paths)
        self.assertIn("/research", paths)

    def test_secret_flows_through_settings(self) -> None:
        from src.config import Settings

        with patch.dict(os.environ, {"MCP_JWT_SECRET": SECRET}):
            self.assertEqual(Settings().mcp_jwt_secret, SECRET)

    def test_unset_secret_warns_at_startup_without_breaking_api(self) -> None:
        import server

        fake_settings = SimpleNamespace(
            run_cost_cap_warning=None,
            mcp_jwt_secret=None,
        )
        stdout = StringIO()
        with (
            patch.object(server, "settings", fake_settings),
            patch.object(server, "init_db"),
            redirect_stdout(stdout),
        ):
            with TestClient(server.app) as client:
                self.assertEqual(client.get("/ping").json(), {"ok": True})
        self.assertIn("WARNING", stdout.getvalue())
        self.assertIn("MCP_JWT_SECRET is unset", stdout.getvalue())

    def test_mcp_module_cannot_import_pipeline_execution_layers(self) -> None:
        source = (
            Path(__file__).resolve().parent.parent / "src" / "mcp_server.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("src.graph", "src.agents", "src.tools", "run_research"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
