"""Evaluation harness for accuracy, faithfulness, cost, and latency.

Runs a labelled golden set through the full pipeline and reports a metrics table.
Accuracy and faithfulness are both LLM-as-judged; cost comes from the real token
usage LiteLLM reports per run; latency is wall-clock end-to-end.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sys
from pathlib import Path as _Path

# Ensure repo root is importable so `src` resolves regardless of CWD.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.llm import chat  # noqa: E402
from src.schemas import ResearchReport  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Fallback per-token cost for providers that don't price (e.g. local Ollama).
TOKEN_COST_PER_1K = 0.0005


@dataclass
class EvalSample:
    question: str
    expected_keywords: list[str]  # ground-truth terms the judge scores against


@dataclass
class EvalResult:
    question: str
    accuracy: float
    faithfulness: float
    cost_usd: float
    latency_s: float
    tokens: int = 0
    summary: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    def table(self) -> str:
        rows = [
            "| Question | Accuracy | Faithfulness | Cost ($) | Tokens | Latency (s) |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            rows.append(
                f"| {r.question[:40]} | {r.accuracy:.2f} | {r.faithfulness:.2f} "
                f"| {r.cost_usd:.4f} | {r.tokens} | {r.latency_s:.2f} |"
            )
        if self.results:
            n = len(self.results)
            avg = lambda k: sum(getattr(r, k) for r in self.results) / n  # noqa: E731
            rows.append(
                f"| **Avg** | {avg('accuracy'):.2f} | {avg('faithfulness'):.2f} "
                f"| {avg('cost_usd'):.4f} | {int(avg('tokens'))} | {avg('latency_s'):.2f} |"
            )
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return {"results": [asdict(r) for r in self.results]}

    def save(self, name: str = "latest") -> Path:
        path = REPORTS_DIR / f"{name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, name: str) -> "EvalReport":
        path = REPORTS_DIR / f"{name}.json"
        data = json.loads(path.read_text())
        return cls(results=[EvalResult(**r) for r in data["results"]])


def _accuracy(report: ResearchReport, sample: EvalSample) -> float:
    """LLM-as-judge: how well the report covers the ground-truth keywords."""
    keywords = ", ".join(sample.expected_keywords)
    prompt = (
        "You are grading a research report. Does the report substantively "
        "cover each of these key concepts? Answer with a single number: the "
        "fraction of concepts covered.\n\n"
        f"KEY CONCEPTS:\n{keywords}\n\n"
        f"REPORT:\n{report.summary}\n\nFINDINGS:\n"
        + "\n".join(f"- {f}" for f in report.findings)
    )
    try:
        raw = chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
        import re

        m = re.search(r"\d+(\.\d+)?", raw)
        val = float(m.group()) if m else 0.5
        return max(0.0, min(1.0, val))
    except Exception:  # noqa: BLE001
        # Fallback to lexical coverage if the judge is unavailable.
        text = f"{report.summary}\n" + "\n".join(report.findings)
        found = sum(1 for kw in sample.expected_keywords if kw.lower() in text.lower())
        return found / len(sample.expected_keywords)


def _faithfulness(report: ResearchReport) -> float:
    """LLM-as-judge: fraction of findings supported by the cited sources."""
    if not report.sources:
        return 0.0
    corpus = "\n".join(s.snippet for s in report.sources)[:8000]
    prompt = (
        "Rate faithfulness. For each finding below, state whether it is SUPPORTED "
        "or UNSUPPORTED by the given source snippets. Answer with a single number: "
        "the fraction of findings that are SUPPORTED.\n\n"
        f"FINDINGS:\n{chr(10).join('- ' + f for f in report.findings)}\n\n"
        f"SOURCES:\n{corpus}"
    )
    try:
        raw = chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
        import re

        m = re.search(r"\d+(\.\d+)?", raw)
        val = float(m.group()) if m else 0.5
        return max(0.0, min(1.0, val))
    except Exception:  # noqa: BLE001
        return 0.5


def evaluate(sample: EvalSample, run_fn) -> EvalResult:
    """Run one sample through a provided run_fn(question)->ResearchReport."""
    start = time.time()
    report = run_fn(sample.question)
    latency = time.time() - start
    # Real cost from the run's LiteLLM usage; estimate only when the provider
    # doesn't price (local Ollama returns cost 0).
    cost = report.cost_usd or (report.total_tokens / 1000) * TOKEN_COST_PER_1K
    return EvalResult(
        question=sample.question,
        accuracy=_accuracy(report, sample),
        faithfulness=_faithfulness(report),
        cost_usd=round(cost, 6),
        latency_s=round(latency, 2),
        tokens=report.total_tokens,
        summary=report.summary[:200],
    )


def run_harness(samples: list[EvalSample], run_fn, *, name: str = "latest") -> EvalReport:
    report = EvalReport()
    for sample in samples:
        report.results.append(evaluate(sample, run_fn))
    path = report.save(name)
    print(report.table())
    print(f"\nSaved to: {path}")
    return report


def compare(name_a: str, name_b: str) -> str:
    """Side-by-side diff of two saved eval reports (e.g. prompt/model changes)."""
    a = EvalReport.load(name_a)
    b = EvalReport.load(name_b)
    rows = [
        "| Question | Metric | A | B | Δ |",
        "|---|---|---|---|---|",
    ]
    for ra, rb in zip(a.results, b.results):
        for metric in ("accuracy", "faithfulness", "cost_usd", "latency_s", "tokens"):
            va, vb = getattr(ra, metric), getattr(rb, metric)
            d = vb - va
            rows.append(f"| {ra.question[:32]} | {metric} | {va:.4f} | {vb:.4f} | {d:+.4f} |")
    table = "\n".join(rows)
    print(table)
    return table


def run_fn_from_cli(question: str) -> ResearchReport:
    """Default run_fn: the full pipeline."""
    from src.graph import run_research

    return run_research(question)