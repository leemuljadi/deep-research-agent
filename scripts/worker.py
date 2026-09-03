"""Worker entrypoint: claim jobs from the Postgres queue and execute the graph.

Usage:
    python -m scripts.worker
"""
from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import (  # noqa: E402
    claim_next_job,
    complete_job,
    fail_job,
    init_db,
    snapshot_state,
)
from src.graph import (  # noqa: E402
    deserialize_state,
    initial_state,
    run_research_state,
    serialize_state,
)


def _poll_seconds() -> float:
    try:
        return float(os.getenv("WORKER_POLL_SECONDS", "2"))
    except ValueError:
        return 2.0  # bad env value: fall back to the documented default


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def process_one_job(worker_id: str) -> bool:
    """Claim and run one job; True when a job was processed, False on empty queue."""
    try:
        job = claim_next_job(worker_id)
    except Exception as exc:
        # DB outage during claim: never kill the poll loop (spec I/O matrix).
        print(f"[worker {worker_id}] claim failed ({type(exc).__name__}: {exc}); retrying")
        return False
    if job is None:
        return False
    job_id = str(job["id"])
    print(f"[worker {worker_id}] claimed job {job_id}: {job['question']!r}")
    try:
        # Resume path (AD-14): when the row carries a state snapshot, rebuild
        # the graph state solely from Postgres — never from worker memory.
        if job.get("state_snapshot"):
            try:
                state = deserialize_state(job["state_snapshot"])
            except ValueError as exc:
                fail_job(job_id, f"ValueError: corrupt snapshot: {exc}")
                print(f"[worker {worker_id}] job {job_id} failed: {exc}")
                return True
            # A redirect decision re-points the run at a new question: the
            # prior plan and its sub-results no longer answer it, so they are
            # cleared and every gate re-arms — the resumed run re-plans against
            # the redirected question and the human re-approves each gate.
            if job.get("resume_question"):
                state["question"] = job["resume_question"]
                state["plan"] = None
                state["sub_results"] = []
                state["passed_gates"] = []
            gates = list(state.get("gate_policy") or [])
        else:
            gates = _gate_policy(job)
            state = initial_state(job["question"], gates)

        state, gate = run_research_state(state, gates)
        if gate is not None:
            # Gate hit: snapshot the full state to Postgres, flip the row to
            # `waiting_for_input` and exit the claim — the job is NOT complete.
            snapshot_state(job_id, serialize_state(state))
            print(f"[worker {worker_id}] job {job_id} paused at gate {gate!r}")
            return True

        report = state["report"]
        complete_job(job_id, report.model_dump())
        print(f"[worker {worker_id}] job {job_id} completed")
    except Exception as exc:
        try:
            fail_job(job_id, f"{type(exc).__name__}: {exc}")
        except Exception as fail_exc:
            # Recording the failure failed (db outage); the row stays 'running'
            # per the frozen leave-visible policy. Never crash the loop.
            print(f"[worker {worker_id}] job {job_id} FAILED but recording failed too: {fail_exc}")
        print(f"[worker {worker_id}] job {job_id} failed: {exc}")
    return True


def _gate_policy(job: dict) -> list[str]:
    """Per-run gate policy off the job row; guard against bad env/data.

    Non-list data AND all-unknown names fall back to the DDL default — a
    corrupt policy must not silently disable human oversight.
    """
    policy = job.get("gate_policy")
    known = ("plan", "synthesis")
    if not isinstance(policy, list):
        return ["plan", "synthesis"]  # the DDL default
    filtered = [g for g in policy if g in known]
    if not filtered and policy:  # well-formed list, but nothing recognizable
        return ["plan", "synthesis"]
    return filtered


def main() -> None:
    init_db()
    worker_id = _worker_id()
    poll = _poll_seconds()
    print(f"[worker {worker_id}] polling for jobs every {poll}s")
    while True:
        if not process_one_job(worker_id):
            time.sleep(poll)


if __name__ == "__main__":
    main()