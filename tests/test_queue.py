from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src import db
from src.schemas import RunStatus, RunStatusResponse
from unittest.mock import MagicMock, patch


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


if __name__ == "__main__":
    unittest.main()