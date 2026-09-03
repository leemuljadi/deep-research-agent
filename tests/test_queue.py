from __future__ import annotations

import contextlib
import json
import unittest
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from src import db, graph
from src.llm import Usage
from src.schemas import (
    ResearchPlan,
    ResearchReport,
    RunStatus,
    RunStatusResponse,
    SubQuestionResult,
)


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


def _waiting_row(**overrides) -> dict:
    """A row shaped like a `waiting_for_input` research_jobs row."""
    row = {
        "id": "d0a1b2c3-d4e5-4678-9abc-def012345678",
        "question": "What is RAG?",
        "status": "waiting_for_input",
        "worker_id": None,
        "gate_policy": ["plan", "synthesis"],
        "state_snapshot": {"question": "What is RAG?", "passed_gates": ["plan"]},
        "resume_question": None,
        "decisions": [],
        "result": None,
        "error": None,
        "created_at": datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 9, 3, 12, 5, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row



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

        job = {"id": "job-1", "question": "q", "gate_policy": []}
        report = MagicMock()
        report.model_dump.return_value = {"summary": "s"}
        state = {"report": report, "gate_policy": [], "passed_gates": []}
        with (
            patch.object(worker, "claim_next_job", return_value=job) as claim,
            patch.object(worker, "run_research_state", return_value=(state, None)) as run,
            patch.object(worker, "complete_job") as complete,
            patch.object(worker, "fail_job") as fail,
        ):
            self.assertTrue(worker.process_one_job("host:1"))
        claim.assert_called_once_with("host:1")
        run.assert_called_once()
        complete.assert_called_once()
        fail.assert_not_called()

    def test_process_one_job_failure_lands_failed_and_keeps_going(self) -> None:
        from scripts import worker

        job = {"id": "job-1", "question": "q", "gate_policy": []}
        with (
            patch.object(worker, "claim_next_job", return_value=job),
            patch.object(
                worker, "run_research_state", side_effect=ValueError("parse fail")
            ),
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

    def test_cost_cap_row_maps_error_with_summary(self) -> None:
        response = self._get(
            self._row(
                status="cost_cap_exceeded",
                error="cost cap exceeded for run abc: accumulated $1.20 >= cap $1.00",
            )
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "cost_cap_exceeded")
        self.assertIn("cost cap exceeded", body["error"])
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


class CostCapTests(unittest.TestCase):
    """Run-scoped cap accumulator (AD-15): the one enforcement home."""

    def setUp(self) -> None:
        # Each test starts with no run token; any leaked scope fails loudly.
        import src.llm as llm

        self.llm = llm
        if llm.current_run_id() is not None:
            llm.clear_run_cap(llm.current_run_id())

    def test_accumulator_crosses_at_one_call_overshoot(self) -> None:
        """Cap 1.0, captures summing past it: the capture that crosses raises
        with run id + cap + total; earlier captures pass untouched."""
        self.llm.set_run_cap("run-1", 1.0)
        self.llm.register_usage("run-1", Usage(cost_usd=0.40))
        self.llm.register_usage("run-1", Usage(cost_usd=0.42))
        with self.assertRaises(self.llm.CostCapExceeded) as ctx:
            self.llm.register_usage("run-1", Usage(cost_usd=0.30))
        msg = str(ctx.exception)
        self.assertIn("run-1", msg)
        self.assertIn("1.0", msg)
        self.assertIn("1.12", msg)
        # Overshoot bound: the second call's own cost never landed.
        with self.assertRaises(self.llm.CostCapExceeded):
            self.llm.check_cost("run-1", 0.0)

    def test_uncapped_path_never_raises(self) -> None:
        """No scope → capture is a no-op; behavior identical to story 4."""
        for _ in range(5):
            self.llm.register_usage(None, Usage(cost_usd=99.0))

    def test_scope_registered_but_cap_none_never_raises(self) -> None:
        """set_run_cap(id, None): the run token exists but the cap is None."""
        self.llm.set_run_cap("run-u", None)
        self.assertIsNone(self.llm.get_run_cap())
        self.llm.register_usage("run-u", Usage(cost_usd=42.0))

    def test_cap_exactly_reached_trips_ge_semantics(self) -> None:
        self.llm.set_run_cap("run-e", 1.0)
        self.llm.register_usage("run-e", Usage(cost_usd=0.6))
        with self.assertRaises(self.llm.CostCapExceeded):
            self.llm.register_usage("run-e", Usage(cost_usd=0.4))

    def test_zero_cost_run_never_trips_any_cap(self) -> None:
        self.llm.set_run_cap("run-z", 0.01)
        for _ in range(10):
            self.llm.register_usage("run-z", Usage(total_tokens=100, cost_usd=0.0))

    def test_zero_cap_trips_on_first_cost_bearing_call(self) -> None:
        self.llm.set_run_cap("run-0", 0.0)
        self.llm.register_usage("run-0", Usage(cost_usd=0.0))  # free: no trip
        with self.assertRaises(self.llm.CostCapExceeded):
            self.llm.register_usage("run-0", Usage(cost_usd=0.001))

    def test_seed_run_spend_continues_after_gate_resume(self) -> None:
        """A resumed segment re-seeds from the snapshot's usage_log sum, then
        the next capture checks against the seeded total."""
        self.llm.set_run_cap("run-r", 1.0)
        self.llm.seed_run_spend(0.9)
        self.llm.register_usage("run-r", Usage(cost_usd=0.05))
        with self.assertRaises(self.llm.CostCapExceeded):
            self.llm.register_usage("run-r", Usage(cost_usd=0.10))  # 1.05 >= 1.0

    def test_seed_without_scope_is_noop(self) -> None:
        """Reseeding with no active scope (worker finally ran, etc.) no-ops."""
        self.llm.seed_run_spend(99.0)

    def test_clear_run_cap_drops_scope_and_token(self) -> None:
        self.llm.set_run_cap("run-c", 1.0)
        self.llm.clear_run_cap("run-c")
        self.assertIsNone(self.llm.current_run_id())
        self.assertIsNone(self.llm.get_run_cap())
        self.llm.register_usage("run-c", Usage(cost_usd=99.0))

    def test_one_call_overshoot_leaves_partial_usage_counted(self) -> None:
        """The tripping capture's cost IS folded in (total reflects the last
        in-flight call) — the raise happens after the fold, before return."""
        self.llm.set_run_cap("run-o", 1.0)
        self.llm.register_usage("run-o", Usage(cost_usd=0.6))
        with self.assertRaises(self.llm.CostCapExceeded) as ctx:
            self.llm.register_usage("run-o", Usage(cost_usd=0.9))
        # 0.6 + 0.9 = 1.5: the tripping call's cost landed in the total.
        self.assertIn("1.5", str(ctx.exception))

    def test_chat_with_usage_checks_current_run_cap(self) -> None:
        """Capture path: chat usage folds into the scoped run and trips there."""
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            _hidden_params={"response_cost": 0.75},
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )
        self.llm.set_run_cap("run-chat", 1.0)
        with patch.object(self.llm, "get_router") as router:
            router.return_value.completion.return_value = resp
            _, u1 = self.llm.chat_with_usage([{"role": "user", "content": "q"}])
            self.assertEqual(u1.cost_usd, 0.75)
            with self.assertRaises(self.llm.CostCapExceeded):
                self.llm.chat_with_usage([{"role": "user", "content": "q"}])

    def test_embed_texts_checks_current_run_cap(self) -> None:
        class _EmbResp(dict):  # litellm response: subscript + hidden params
            _hidden_params = {"response_cost": 0.6}

        self.llm.set_run_cap("run-emb", 0.5)
        with patch.object(self.llm, "get_router") as router:
            router.return_value.embedding.return_value = _EmbResp(
                data=[{"embedding": [0.1]}]
            )
            # Cost 0.6 >= cap 0.5: the capture itself trips before returning.
            with self.assertRaises(self.llm.CostCapExceeded):
                self.llm.embed_texts(["hello"])


class WorkerCapTests(unittest.TestCase):
    """Worker-side cap wiring (AD-15): set/seed/clear + cost_cap_exceeded row."""

    def _claim_row(self, **overrides) -> dict:
        row = {
            "id": "job-1",
            "question": "q",
            "gate_policy": [],
            "state_snapshot": None,
            "resume_question": None,
            "cost_cap_usd": None,
        }
        row.update(overrides)
        return row

    def test_worker_maps_cost_cap_exceeded_to_status(self) -> None:
        from scripts import worker
        from src.llm import CostCapExceeded

        with (
            patch.object(worker, "claim_next_job", return_value=self._claim_row()),
            patch.object(
                worker, "run_research_state", side_effect=CostCapExceeded(
                    "cost cap exceeded for run job-1: accumulated $1.0500 >= cap $1.0000"
                )
            ),
            patch.object(worker, "cost_cap_job") as cap_job,
            patch.object(worker, "fail_job") as fail,
            patch.object(worker, "complete_job") as complete,
            patch.object(worker.llm, "set_run_cap") as set_cap,
            patch.object(worker.llm, "clear_run_cap") as clear_cap,
        ):
            # Survives: the worker moves on to subsequent jobs.
            self.assertTrue(worker.process_one_job("host:1"))
        cap_job.assert_called_once_with("job-1", "cost cap exceeded for run job-1: accumulated $1.0500 >= cap $1.0000")
        fail.assert_not_called()
        complete.assert_not_called()
        set_cap.assert_called_once_with("job-1", None)  # no cap anywhere: uncapped
        clear_cap.assert_called_once_with("job-1")

    def test_worker_prefers_row_cap_over_env(self) -> None:
        from scripts import worker

        with (
            patch.object(worker, "claim_next_job", return_value=self._claim_row(cost_cap_usd=0.25)),
            patch.object(worker, "run_research_state", return_value=({"report": None}, "plan")),
            patch.object(worker, "snapshot_state"),
            patch.object(worker.llm, "set_run_cap") as set_cap,
            patch.object(worker.llm, "seed_run_spend") as seed,
            patch.object(worker.llm, "clear_run_cap"),
        ):
            worker.process_one_job("host:1")
        set_cap.assert_called_once_with("job-1", 0.25)
        seed.assert_called_once_with(0.0)

    def test_worker_falls_back_to_env_cap(self) -> None:
        from scripts import worker

        with (
            patch.object(worker, "claim_next_job", return_value=self._claim_row()),
            patch.object(worker, "run_research_state", return_value=({"report": None}, "plan")),
            patch.object(worker, "snapshot_state"),
            patch.object(worker, "settings") as settings,
            patch.object(worker.llm, "set_run_cap") as set_cap,
            patch.object(worker.llm, "clear_run_cap"),
        ):
            settings.run_cost_cap_usd = 2.0
            worker.process_one_job("host:1")
        set_cap.assert_called_once_with("job-1", 2.0)

    def test_worker_seeds_accumulator_from_snapshot_usage(self) -> None:
        from scripts import worker

        snapshot_usage = [Usage(cost_usd=0.30), Usage(cost_usd=0.12)]
        state = {
            "report": None,
            "usage_log": snapshot_usage,
            "gate_policy": [],
            "passed_gates": [],
        }
        with (
            patch.object(
                worker,
                "claim_next_job",
                return_value=self._claim_row(state_snapshot={"q": 1}),
            ),
            patch.object(worker, "deserialize_state", return_value=state),
            patch.object(worker, "run_research_state", return_value=(state, None)),
            patch.object(worker.llm, "seed_run_spend") as seed,
            patch.object(worker.llm, "set_run_cap"),
            patch.object(worker.llm, "clear_run_cap"),
        ):
            worker.process_one_job("host:1")
        seed.assert_called_once_with(0.42)

    def test_worker_clears_cap_in_finally_on_generic_failure(self) -> None:
        from scripts import worker

        with (
            patch.object(worker, "claim_next_job", return_value=self._claim_row()),
            patch.object(worker, "run_research_state", side_effect=ValueError("boom")),
            patch.object(worker, "fail_job"),
            patch.object(worker.llm, "set_run_cap") as set_cap,
            patch.object(worker.llm, "clear_run_cap") as clear_cap,
        ):
            worker.process_one_job("host:1")
        set_cap.assert_called_once()
        clear_cap.assert_called_once_with("job-1")


class CostCapDdlTests(unittest.TestCase):
    """cost_cap_usd column: DDL + additive migration (AD-15)."""

    def test_init_db_declares_cost_cap_usd_column(self) -> None:
        cursor = FakeCursor()
        with _patched(cursor):
            db.init_db()
        sql = _joined_sql(cursor)
        self.assertIn("cost_cap_usd DOUBLE PRECISION NULL", sql)
        self.assertIn("cost_cap_usd DOUBLE PRECISION", sql)  # migration ALTER
        self.assertIn("ADD COLUMN IF NOT EXISTS cost_cap_usd", sql)

    def test_cost_cap_job_mirrors_fail_job(self) -> None:
        cursor = FakeCursor()
        with _patched(cursor):
            db.cost_cap_job("job-1", "cost cap exceeded: $1.05 >= $1.00")
        self.assertEqual(len(cursor.calls), 1)
        sql, params = cursor.calls[0]
        self.assertIn("SET status = 'cost_cap_exceeded', error = %s", sql)
        self.assertEqual(params, ("cost cap exceeded: $1.05 >= $1.00", "job-1"))


class GateDdlTests(unittest.TestCase):
    def test_init_db_declares_gate_columns_with_defaults(self) -> None:
        cursor = FakeCursor()
        with _patched(cursor):
            db.init_db()
        sql = _joined_sql(cursor)
        self.assertIn("gate_policy JSONB NOT NULL DEFAULT '[\"plan\",\"synthesis\"]'", sql)
        self.assertIn("state_snapshot JSONB", sql)
        self.assertIn("resume_question TEXT", sql)
        self.assertIn("decisions   JSONB NOT NULL DEFAULT '[]'", sql)


class ConditionalTransitionTests(unittest.TestCase):
    """Api-owned writes are conditional single-statement UPDATEs (AD-14)."""

    def test_resume_run_is_conditional_single_statement(self) -> None:
        cursor = FakeCursor(rows=[{"id": "job-1"}])
        with _patched(cursor):
            self.assertTrue(db.resume_run("job-1"))
        self.assertEqual(len(cursor.calls), 1)
        sql, params = cursor.calls[0]
        self.assertIn("SET status = 'queued'", sql)
        self.assertIn("WHERE id = %s AND status = 'waiting_for_input'", sql)
        self.assertIn("RETURNING id", sql)
        self.assertEqual(params, (None, "job-1"))

    def test_resume_run_zero_rows_returns_false(self) -> None:
        """Race simulation: another actor resumed first → 0 rows → False → 409."""
        cursor = FakeCursor()  # fetchone returns None: no rows matched
        with _patched(cursor):
            self.assertFalse(db.resume_run("job-1"))

    def test_resume_run_carries_resume_question_for_redirect(self) -> None:
        cursor = FakeCursor(rows=[{"id": "job-1"}])
        with _patched(cursor):
            self.assertTrue(db.resume_run("job-1", resume_question="New question"))
        sql, params = cursor.calls[0]
        self.assertIn("resume_question = %s", sql)
        self.assertEqual(params, ("New question", "job-1"))

    def test_cancel_run_covers_waiting_and_queued_in_one_statement(self) -> None:
        cursor = FakeCursor(rows=[{"id": "job-1"}])
        with _patched(cursor):
            self.assertTrue(db.cancel_run("job-1"))
        self.assertEqual(len(cursor.calls), 1)
        sql, params = cursor.calls[0]
        self.assertIn("SET status = 'cancelled'", sql)
        self.assertIn("status IN ('waiting_for_input', 'queued')", sql)
        self.assertNotIn("'running'", sql)
        self.assertEqual(params, ("job-1",))

    def test_cancel_running_row_touches_zero_rows(self) -> None:
        """A running row is worker-owned; the api's cancel must not match it."""
        cursor = FakeCursor()
        with _patched(cursor):
            self.assertFalse(db.cancel_run("job-1"))

    def test_record_decision_appends_conditionally(self) -> None:
        cursor = FakeCursor(rows=[{"id": "job-1"}])
        with _patched(cursor):
            self.assertTrue(db.record_decision("job-1", "approve", {"k": "v"}))
        sql, params = cursor.calls[0]
        self.assertIn("decisions = decisions || %s", sql)
        self.assertIn("AND status = 'waiting_for_input'", sql)
        payload = params[0].obj
        self.assertEqual(payload["decision"], "approve")
        self.assertEqual(payload["payload"], {"k": "v"})
        self.assertIn("decided_at", payload)

    def test_snapshot_state_is_unconditional_worker_write(self) -> None:
        """The worker owns claimed rows — its pause write has no status guard."""
        cursor = FakeCursor()
        with _patched(cursor):
            db.snapshot_state("job-1", {"question": "q"})
        sql, params = cursor.calls[0]
        self.assertIn("SET status = 'waiting_for_input'", sql)
        self.assertIn("state_snapshot = %s", sql)
        self.assertIn("worker_id = NULL", sql)
        self.assertNotIn("AND status", sql)
        self.assertEqual(params[1], "job-1")


class GateEndpointTests(unittest.TestCase):
    """POST /runs/{id}/approve|redirect|cancel: thin shells over the writes."""

    UUID = "d0a1b2c3-d4e5-4678-9abc-def012345678"

    def _post(self, path: str, json_body=None) -> object:
        import server

        return TestClient(server.app).post(f"/runs/{self.UUID}/{path}", json=json_body)

    def test_approve_flips_waiting_run_to_queued(self) -> None:
        import server

        with (
            patch.object(server, "get_job", return_value=_waiting_row()),
            patch.object(server, "record_decision", return_value=True) as rec,
            patch.object(server, "resume_run", return_value=True) as res,
        ):
            response = self._post("approve")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"run_id": self.UUID, "status": "queued"})
        rec.assert_called_once_with(self.UUID, "approve", {})
        res.assert_called_once_with(self.UUID)

    def test_approve_on_unknown_id_is_404(self) -> None:
        import server

        with patch.object(server, "get_job", return_value=None):
            response = self._post("approve")
        self.assertEqual(response.status_code, 404)

    def test_double_approve_is_409(self) -> None:
        """Second approve after resume: the conditional UPDATE matched 0 rows."""
        import server

        with (
            patch.object(server, "get_job", return_value=_waiting_row()),
            patch.object(server, "record_decision", return_value=False),
        ):
            response = self._post("approve")
        self.assertEqual(response.status_code, 409)

    def test_approve_race_between_decision_and_resume_is_409(self) -> None:
        import server

        with (
            patch.object(server, "get_job", return_value=_waiting_row()),
            patch.object(server, "record_decision", return_value=True),
            patch.object(server, "resume_run", return_value=False),
        ):
            response = self._post("approve")
        self.assertEqual(response.status_code, 409)

    def test_redirect_requires_question_payload(self) -> None:
        import server

        with patch.object(server, "get_job", return_value=_waiting_row()):
            response = self._post("redirect", {"decision": "redirect", "payload": {}})
        self.assertEqual(response.status_code, 422)

    def test_redirect_re_enqueues_with_new_question(self) -> None:
        import server

        with (
            patch.object(server, "get_job", return_value=_waiting_row()),
            patch.object(server, "record_decision", return_value=True) as rec,
            patch.object(server, "resume_run", return_value=True) as res,
        ):
            response = self._post(
                "redirect",
                {"decision": "redirect", "payload": {"question": "  New Q  "}},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question"], "New Q")
        rec.assert_called_once_with(self.UUID, "redirect", {"question": "  New Q  "})
        res.assert_called_once_with(self.UUID, resume_question="New Q")

    def test_redirect_decision_shape_is_validated(self) -> None:
        """GateDecision rejects unknown decisions and non-dict payloads (AD-6)."""
        from src.schemas import GateDecision

        GateDecision(decision="approve", payload={})
        with self.assertRaises(Exception):
            GateDecision(decision="cancel", payload={})

    def test_cancel_waiting_run_succeeds(self) -> None:
        import server

        with (
            patch.object(server, "get_job", return_value=_waiting_row()),
            patch.object(server, "cancel_run", return_value=True) as cancel,
        ):
            response = self._post("cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"run_id": self.UUID, "status": "cancelled"})
        cancel.assert_called_once_with(self.UUID)

    def test_cancel_running_run_is_409(self) -> None:
        import server

        running = _waiting_row(status="running", worker_id="host:1")
        with (
            patch.object(server, "get_job", return_value=running),
            patch.object(server, "cancel_run", return_value=False),
        ):
            response = self._post("cancel")
        self.assertEqual(response.status_code, 409)

    def test_gate_endpoints_404_on_unknown_id(self) -> None:
        import server

        for verb in ("approve", "redirect", "cancel"):
            with self.subTest(verb=verb):
                with patch.object(server, "get_job", return_value=None):
                    body = (
                        {"decision": "redirect", "payload": {"question": "q"}}
                        if verb == "redirect"
                        else None
                    )
                    response = self._post(verb, body)
                    self.assertEqual(response.status_code, 404)


class WorkerGateTests(unittest.TestCase):
    """Worker pause at a gate + resume rebuild (AD-14 worker side)."""

    def _claim_row(self, **overrides) -> dict:
        row = {
            "id": "job-1",
            "question": "q",
            "gate_policy": ["plan", "synthesis"],
            "state_snapshot": None,
            "resume_question": None,
        }
        row.update(overrides)
        return row

    def test_worker_pauses_at_plan_gate_and_snapshots(self) -> None:
        from scripts import worker
        from src.graph import run_research_state

        paused_state = {
            "question": "q",
            "plan": ResearchPlan(objective="o", sub_questions=["s"]),
            "sub_results": [],
            "usage_log": [Usage(total_tokens=10, cost_usd=0.001)],
            "report": None,
            "gate_policy": ["plan", "synthesis"],
            "passed_gates": ["plan"],
        }
        with (
            patch.object(worker, "claim_next_job", return_value=self._claim_row()),
            patch.object(
                worker, "run_research_state", return_value=(paused_state, "plan")
            ) as run,
            patch.object(worker, "snapshot_state") as snap,
            patch.object(worker, "complete_job") as complete,
        ):
            self.assertTrue(worker.process_one_job("host:1"))
        snap.assert_called_once()
        job_id, snapshot = snap.call_args[0]
        self.assertEqual(job_id, "job-1")
        # The snapshot round-trips the usage_log and passed gate.
        self.assertEqual(snapshot["usage_log"], [{"total_tokens": 10, "prompt_tokens": 0,
                                                  "completion_tokens": 0, "cost_usd": 0.001}])
        self.assertEqual(snapshot["passed_gates"], ["plan"])
        complete.assert_not_called()  # paused, NOT completed

    def test_worker_resume_rebuilds_from_snapshot_and_completes(self) -> None:
        from scripts import worker

        report = MagicMock()
        report.model_dump.return_value = {"summary": "s"}
        rebuilt = {
            "question": "q",
            "plan": None,
            "sub_results": [],
            "usage_log": [],
            "report": report,
            "gate_policy": ["plan", "synthesis"],
            "passed_gates": ["plan", "synthesis"],
        }
        with (
            patch.object(
                worker,
                "claim_next_job",
                return_value=self._claim_row(
                    state_snapshot={"question": "q", "passed_gates": ["plan"]}
                ),
            ),
            patch.object(worker, "deserialize_state", return_value=rebuilt),
            patch.object(
                worker, "run_research_state", return_value=(rebuilt, None)
            ),
            patch.object(worker, "complete_job") as complete,
            patch.object(worker, "snapshot_state") as snap,
        ):
            self.assertTrue(worker.process_one_job("host:1"))
        complete.assert_called_once_with("job-1", {"summary": "s"})
        snap.assert_not_called()

    def test_worker_resume_applies_redirect_question(self) -> None:
        from scripts import worker

        rebuilt = {
            "question": "old",
            "plan": "stale-plan",
            "sub_results": ["stale"],
            "gate_policy": ["plan"],
            "passed_gates": ["plan"],
        }
        with (
            patch.object(
                worker,
                "claim_next_job",
                return_value=self._claim_row(
                    state_snapshot={"question": "old", "passed_gates": ["plan"]},
                    resume_question="new question",
                ),
            ),
            patch.object(worker, "deserialize_state", return_value=rebuilt) as deser,
            patch.object(
                worker,
                "run_research_state",
                side_effect=lambda s, g: (s, None),
            ) as run,
            patch.object(worker, "complete_job") as complete,
        ):
            self.assertTrue(worker.process_one_job("host:1"))
        # The redirect question replaced the snapshot's question, the stale
        # plan/sub-results were cleared, and the plan gate re-arms so the
        # resumed run plans against the redirected question.
        self.assertEqual(rebuilt["question"], "new question")
        self.assertIsNone(rebuilt["plan"])
        self.assertEqual(rebuilt["sub_results"], [])
        self.assertEqual(rebuilt["passed_gates"], [])
        run.assert_called_once()

    def test_worker_corrupt_snapshot_fails_loudly(self) -> None:
        from scripts import worker

        with (
            patch.object(
                worker,
                "claim_next_job",
                return_value=self._claim_row(state_snapshot={"plan": "garbage"}),
            ),
            patch.object(
                worker,
                "deserialize_state",
                side_effect=ValueError("corrupt state snapshot: bad plan"),
            ),
            patch.object(worker, "fail_job") as fail,
            patch.object(worker, "complete_job") as complete,
        ):
            self.assertTrue(worker.process_one_job("host:1"))
        fail.assert_called_once()
        self.assertIn("corrupt", fail.call_args[0][1])
        complete.assert_not_called()

    def test_worker_bad_gate_policy_falls_back_to_default(self) -> None:
        from scripts import worker

        self.assertEqual(
            worker._gate_policy({"gate_policy": "not-a-list"}), ["plan", "synthesis"]
        )
        self.assertEqual(
            worker._gate_policy({"gate_policy": ["plan", "bogus"]}), ["plan"]
        )


class GatedRunnerTests(unittest.TestCase):
    """run_research_state: the single pause construct, policy as data (AD-14)."""

    def _patches(self, plan_qs=("sub q",)):
        plan = ResearchPlan(objective="o", sub_questions=list(plan_qs))
        report = ResearchReport(
            summary="s", findings=[], sources=[], confidence=0.9
        )
        return (
            patch.object(graph, "plan", return_value=(plan, Usage(total_tokens=1))),
            patch.object(
                graph,
                "research_sub_question",
                return_value=(["finding"], [], Usage(total_tokens=2)),
            ),
            patch.object(
                graph, "synthesize", return_value=(report, Usage(total_tokens=3))
            ),
        )

    def test_default_policy_pauses_twice_then_completes(self) -> None:
        with contextlib.ExitStack() as stack:
            for ctx in self._patches():
                stack.enter_context(ctx)
            stack.enter_context(patch.object(graph, "trace", return_value=nullcontext(object())))
            stack.enter_context(patch.object(graph, "record_cost"))

            state, gate = graph.run_research_state(gates=["plan", "synthesis"])
            self.assertEqual(gate, "plan")
            self.assertEqual(state["passed_gates"], ["plan"])
            self.assertIsNotNone(state["plan"])

            # Approve: resume from the snapshot state — planner must NOT re-run.
            with patch.object(graph, "plan", side_effect=AssertionError("re-planned")):
                state, gate = graph.run_research_state(state)
            self.assertEqual(gate, "synthesis")
            self.assertEqual(state["passed_gates"], ["plan", "synthesis"])
            self.assertEqual(state["usage_log"][0].total_tokens, 1)

            # Second approve: run completes past both gates.
            state, gate = graph.run_research_state(state)
            self.assertIsNone(gate)
            self.assertEqual(state["report"].total_tokens, 6)  # 1+2+3 accumulated

    def test_no_gates_completes_without_pausing(self) -> None:
        with contextlib.ExitStack() as stack:
            for ctx in self._patches():
                stack.enter_context(ctx)
            stack.enter_context(patch.object(graph, "trace", return_value=nullcontext(object())))
            stack.enter_context(patch.object(graph, "record_cost"))

            state, gate = graph.run_research_state(gates=[])
            self.assertIsNone(gate)
            self.assertIsNotNone(state["report"])

    def test_snapshot_round_trip_preserves_state(self) -> None:
        state = graph.initial_state("What is RAG?", ["plan", "synthesis"])
        state["plan"] = ResearchPlan(objective="o", sub_questions=["s1", "s2"])
        state["sub_results"] = [
            SubQuestionResult(sub_question="s1", findings=["f"], sources=[])
        ]
        state["usage_log"] = [Usage(prompt_tokens=5, total_tokens=7, cost_usd=0.01)]
        round_tripped = graph.deserialize_state(graph.serialize_state(state))
        self.assertEqual(round_tripped["question"], "What is RAG?")
        self.assertEqual(round_tripped["gate_policy"], ["plan", "synthesis"])
        self.assertEqual(round_tripped["plan"], state["plan"])
        self.assertEqual(round_tripped["sub_results"], state["sub_results"])
        self.assertEqual(round_tripped["usage_log"], state["usage_log"])
        self.assertIsNone(round_tripped["report"])

    def test_snapshot_serialization_is_json_clean(self) -> None:
        state = graph.initial_state("q", ["plan"])
        state["plan"] = ResearchPlan(objective="o", sub_questions=["s"])
        blob = json.dumps(graph.serialize_state(state))
        self.assertIn('"gate_policy"', blob)

    def test_deserialize_corrupt_snapshot_raises_loudly(self) -> None:
        with self.assertRaises(ValueError):
            graph.deserialize_state({"plan": {"objective": 123}})
        with self.assertRaises(ValueError):
            graph.deserialize_state({})  # missing 'question'


if __name__ == "__main__":
    unittest.main()
