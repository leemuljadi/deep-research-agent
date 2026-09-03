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

from src.db import claim_next_job, complete_job, fail_job, init_db  # noqa: E402
from src.graph import run_research  # noqa: E402

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
    question = job["question"]
    print(f"[worker {worker_id}] claimed job {job_id}: {question!r}")
    try:
        report = run_research(question)
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