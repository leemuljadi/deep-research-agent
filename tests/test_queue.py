from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src import db
from src.schemas import RunStatus, RunStatusResponse
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class FakeCursor:
    """Captures (sql, params) per execute; fetchone returns the seeded row."""

    def __init__(self, rows=None) -> None:
        self.calls: list[tuple[str, tuple | None]] = []
        self._rows = list(rows or [])

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patched(cursor: FakeCursor):
    return patch.object(db, "connect", return_value=FakeConn(cursor))


def _joined_sql(cursor: FakeCursor) -> str:
    return "\n".join(sql for sql, _ in cursor.calls)


class JobDdlTests(unittest.TestCase):
    def test_init_db_creates_research_jobs_with_check_and_index(self) -> None:
        cursor = FakeCursor()
        with _patched(cursor):
            db.init_db()
        sql = _joined_sql(cursor)
        self.assertIn("CREATE TABLE IF NOT EXISTS research_jobs", sql)
        self.assertIn("DEFAULT 'queued'", sql)
        self.assertIn("CHECK (status IN", sql)
        self.assertIn("'cost_cap_exceeded'", sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS research_jobs_status_created", sql)


class EnqueueJobTests(unittest.TestCase):
    def test_enqueue_inserts_and_returns_id(self) -> None:
        cursor = FakeCursor(rows=[{"id": "abc-123"}])
        with _patched(cursor):
            job_id = db.enqueue_job("why is the sky blue")
        self.assertEqual(job_id, "abc-123")
        sql, params = cursor.calls[-1]
        self.assertIn("INSERT INTO research_jobs", sql)
        self.assertEqual(params, ("why is the sky blue",))


class ClaimNextJobTests(unittest.TestCase):
    def test_claim_is_single_statement_skip_locked(self) -> None:
        cursor = FakeCursor(rows=[{"id": "job-1", "status": "running"}])
        with _patched(cursor):
            row = db.claim_next_job("host:1")
        self.assertEqual(row["id"], "job-1")
        # Exactly one execute call: the claim is a single statement.
        self.assertEqual(len(cursor.calls), 1)
        sql, params = cursor.calls[0]
        self.assertIn("UPDATE research_jobs", sql)
        self.assertIn("SET status = 'running'", sql)
        self.assertIn("SELECT id FROM research_jobs", sql)
        self.assertIn("FOR UPDATE SKIP LOCKED LIMIT 1", sql)
        self.assertEqual(params, ("host:1",))

    def test_claim_empty_queue_returns_none(self) -> None:
        cursor = FakeCursor()
        with _patched(cursor):
            self.assertIsNone(db.claim_next_job("host:1"))


class TerminalJobTests(unittest.TestCase):
    def test_complete_job_writes_report_json(self) -> None:
        cursor = FakeCursor()
        report = {"summary": "s", "findings": [], "sources": []}
        with _patched(cursor):
            db.complete_job("job-1", report)
        sql, params = cursor.calls[-1]
        self.assertIn("SET status = 'completed', result = %s", sql)
        self.assertEqual(params[1], "job-1")

    def test_fail_job_records_error(self) -> None:
        cursor = FakeCursor()
        with _patched(cursor):
            db.fail_job("job-1", "ValidationError: bad plan")
        sql, params = cursor.calls[-1]
        self.assertIn("SET status = 'failed', error = %s", sql)
        self.assertEqual(params, ("ValidationError: bad plan", "job-1"))


class WorkerLoopTests(unittest.TestCase):
    def test_process_one_job_claims_executes_and_completes(self) -> None:
        from scripts import worker

        job = {"id": "job-1", "question": "q"}
        report = MagicMock()
        report.model_dump.return_value = {"summary": "s"}
        with (
            patch.object(worker, "claim_next_job", return_value=job) as claim,
            patch.object(worker, "run_research", return_value=report) as run,
            patch.object(worker, "complete_job") as complete,
            patch.object(worker, "fail_job") as fail,
        ):
            self.assertTrue(worker.process_one_job("host:1"))
        claim.assert_called_once_with("host:1")
        run.assert_called_once_with("q")
        complete.assert_called_once()
        fail.assert_not_called()

    def test_process_one_job_failure_lands_failed_and_keeps_going(self) -> None:
        from scripts import worker

        job = {"id": "job-1", "question": "q"}
        with (
            patch.object(worker, "claim_next_job", return_value=job),
            patch.object(worker, "run_research", side_effect=ValueError("parse fail")),
            patch.object(worker, "complete_job") as complete,
            patch.object(worker, "fail_job") as fail,
        ):
            # Must not raise — the worker survives per-job failures.
            self.assertTrue(worker.process_one_job("host:1"))
        complete.assert_not_called()
        fail.assert_called_once_with("job-1", "ValueError: parse fail")

    def test_ddl_check_matches_run_status_enum(self) -> None:
        """The DB CHECK membership and the Pydantic enum must not drift apart."""
        cursor = FakeCursor()
        with _patched(cursor):
            db.init_db()
        sql = _joined_sql(cursor)
        check_start = sql.index("CHECK (status IN (")
        check_block = sql[check_start : sql.index("))", check_start)]
        for status in RunStatus:
            self.assertIn(f"'{status.value}'", check_block)

    def test_research_endpoint_returns_run_id_contract(self) -> None:
        """POST /research enqueues and returns only {"run_id": ...} (CAP-1)."""
        import server

        cursor = FakeCursor(rows=[{"id": "abc-123"}])
        with _patched(cursor), patch.object(db, "enqueue_job", return_value="abc-123"):
            response = server.research(server.ResearchIn(question="What is RAG?"))
        self.assertEqual(response, {"run_id": "abc-123"})

    def test_research_endpoint_rejects_blank_question(self) -> None:
        """Blank/whitespace questions 422 before any job row is created."""
        import server
        from pydantic import ValidationError

        for bad in ("", "   "):
            with self.assertRaises(ValidationError):
                server.ResearchIn(question=bad)

    def test_empty_claim_polls_noop(self) -> None:
        from scripts import worker

        with patch.object(worker, "claim_next_job", return_value=None):
            self.assertFalse(worker.process_one_job("host:1"))


class SchemaTests(unittest.TestCase):
    def test_run_status_enum_is_the_closed_set(self) -> None:
        self.assertEqual(
            {s.value for s in RunStatus},
            {
                "queued",
                "running",
                "waiting_for_input",
                "completed",
                "failed",
                "cancelled",
                "cost_cap_exceeded",
            },
        )

    def test_run_status_response_round_trip(self) -> None:
        now = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc).isoformat()
        response = RunStatusResponse(
            run_id="abc-123",
            status=RunStatus.COMPLETED,
            question="q",
            created_at=now,
            updated_at=now,
        )
        data = response.model_dump()
        self.assertEqual(data["status"], RunStatus.COMPLETED)
        restored = RunStatusResponse.model_validate(data)
        self.assertEqual(restored, response)

    def test_completed_response_carries_report(self) -> None:
        now = datetime(2026, 9, 3, tzinfo=timezone.utc).isoformat()
        report = {
            "summary": "s",
            "findings": ["f"],
            "sources": [{"title": "t", "url": None, "snippet": "x"}],
            "confidence": 0.9,
        }
        response = RunStatusResponse(
            run_id="abc-123",
            status=RunStatus.COMPLETED,
            question="q",
            created_at=now,
            updated_at=now,
            report=report,
            error=None,
        )
        self.assertEqual(response.report.summary, "s")


class RunStatusEndpointTests(unittest.TestCase):
    """GET /runs/{run_id}: row → RunStatusResponse mapping and 404s (CAP-1, story 3)."""

    def _row(
        self, *, status: str, result=None, error=None
    ) -> dict:
        return {
            "id": "d0a1b2c3-d4e5-4678-9abc-def012345678",
            "question": "What is RAG?",
            "status": status,
            "result": result,
            "error": error,
            "created_at": datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 9, 3, 12, 5, tzinfo=timezone.utc),
        }

    def _get(self, row) -> object:
        import server

        # No context manager: the app lifespan runs init_db() → real DB;
        # these tests need only the routing layer.
        with patch.object(server, "get_job", return_value=row):
            return TestClient(server.app).get(
                "/runs/d0a1b2c3-d4e5-4678-9abc-def012345678"
            )

    def test_queued_row_maps_without_report_or_error(self) -> None:
        response = self._get(self._row(status="queued"))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["run_id"], "d0a1b2c3-d4e5-4678-9abc-def012345678")
        self.assertEqual(body["question"], "What is RAG?")
        self.assertEqual(body["status"], "queued")
        self.assertIsNone(body["report"])
        self.assertIsNone(body["error"])
        # Row timestamps are datetime → serialized to ISO strings in the mapping.
        self.assertEqual(body["created_at"], "2026-09-03T12:00:00+00:00")
        self.assertEqual(body["updated_at"], "2026-09-03T12:05:00+00:00")

    def test_completed_row_maps_report_through_model_validation(self) -> None:
        report = {
            "summary": "s",
            "findings": ["f1"],
            "sources": [{"title": "t", "url": "https://x", "snippet": "snip"}],
            "confidence": 0.9,
            "cost_usd": 0.01,
            "latency_seconds": 2.5,
            "total_tokens": 100,
        }
        response = self._get(self._row(status="completed", result=report))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertIsNone(body["error"])
        # model_validate normalizes the row JSON (adds Source.score=None).
        self.assertEqual(body["report"]["summary"], "s")
        self.assertEqual(body["report"]["findings"], ["f1"])
        self.assertEqual(
            body["report"]["sources"],
            [{"title": "t", "url": "https://x", "snippet": "snip", "score": None}],
        )
        self.assertEqual(body["report"]["confidence"], 0.9)
        self.assertEqual(body["report"]["total_tokens"], 100)

    def test_failed_row_maps_error_without_report(self) -> None:
        response = self._get(self._row(status="failed", error="ValidationError: bad plan"))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error"], "ValidationError: bad plan")
        self.assertIsNone(body["report"])

    def test_corrupt_result_json_is_500_not_client_error(self) -> None:
        response = self._get(self._row(status="completed", result={"summary": 123}))
        self.assertEqual(response.status_code, 500)
        self.assertIn("failed validation", response.json()["detail"])

    def test_unknown_id_is_404(self) -> None:
        response = self._get(None)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "run not found"})

    def test_malformed_id_is_404_never_500(self) -> None:
        """get_job rejects non-UUID ids before any DB access — no 500, no connection."""
        import server

        for bad in ("../../etc/passwd", "not-a-uuid", "%2e%2e"):
            with self.subTest(bad=bad):
                with patch.object(db, "connect") as never:
                    response = TestClient(server.app).get("/runs/" + bad)
                    never.assert_not_called()
                    self.assertEqual(response.status_code, 404)

    def test_get_job_sql_shape(self) -> None:
        """get_job's SELECT is pinned by shape (house idiom) — schema drift fails here."""
        cursor = FakeCursor(rows=[{"id": "abc", "question": "q"}])
        with _patched(cursor):
            row = db.get_job("d0a1b2c3-d4e5-4678-9abc-def012345678")
        self.assertIsNotNone(row)
        self.assertEqual(
            cursor.calls[0][0], "SELECT * FROM research_jobs WHERE id = %s"
        )
        self.assertEqual(
            cursor.calls[0][1], ("d0a1b2c3-d4e5-4678-9abc-def012345678",)
        )

    def test_get_job_rejects_malformed_id_without_db(self) -> None:
        """Direct unit-level check of the UUID guard (route-level test can be
        satisfied before reaching get_job — this pins the function itself)."""
        with patch.object(db, "connect") as never:
            self.assertIsNone(db.get_job("not-a-uuid"))
            self.assertIsNone(db.get_job("../../etc/passwd"))
            self.assertIsNone(db.get_job(None))
            never.assert_not_called()


if __name__ == "__main__":
    unittest.main()