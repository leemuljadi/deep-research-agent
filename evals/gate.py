"""Tiered eval merge gate (AD-11 v2, story 7): paired candidate-vs-baseline
non-inferiority over the golden-set judge metrics.

Exit codes (the machine contract):
    0  PASS — every pinned metric is non-inferior (or shadow mode: advisory)
    1  BLOCKED — some metric's 95% CI lower bound crossed its tolerance
    3  EVAL_INFRA_FAILURE — malformed judge output, judge timeout, or a
       missing/unreadable report: never counted as a pass or a regression

The gate never imports agent internals — reports are loaded from disk and
the verdict math lives in `evals.eval_harness` (metric-only, no pipeline).

Usage:
    python -m evals.gate <baseline> <candidate>          # manifest enforce flag
    python -m evals.gate --shadow <baseline> <candidate>  # advisory, always exit 0
    python -m evals.gate --help
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Ensure repo root is importable so `evals`/`src` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from evals.eval_harness import (  # noqa: E402
    EvalReport,
    EvalResult,
    GateVerdict,
    JudgeTimeout,
    bootstrap_ci_lower,
    non_inferiority,
)

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.yaml"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

EXIT_PASS = 0
EXIT_BLOCKED = 1
EXIT_EVAL_INFRA_FAILURE = 3


class EvalInfraFailure(Exception):
    """Malformed judge JSON / timeout / missing report — the exit-3 lane."""


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Parse and validate the pinned manifest (spec Code Map: enforce flag,
    trials for pass^k, metrics + tolerances, max_samples for cost control).

    Guarded parsing per the repo convention: a structurally wrong manifest is
    an infra failure, not a silently-defaulted one.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvalInfraFailure(f"manifest unreadable: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise EvalInfraFailure(f"manifest must be a mapping, got {type(data).__name__}")
    for key in ("dataset", "judge", "metrics", "trials", "enforce"):
        if key not in data:
            raise EvalInfraFailure(f"manifest missing required key {key!r}")
    if not isinstance(data["metrics"], dict) or not data["metrics"]:
        raise EvalInfraFailure("manifest 'metrics' must be a non-empty mapping")
    _SUPPORTED_METRICS = ("accuracy", "faithfulness")
    for metric, cfg in data["metrics"].items():
        if metric not in _SUPPORTED_METRICS:
            # A typo'd metric would AttributeError at getattr time and exit 1
            # (blocked) instead of the documented exit 3 — reject it here.
            raise EvalInfraFailure(
                f"metric {metric!r} is not supported "
                f"(supported: {_SUPPORTED_METRICS})"
            )
        if not isinstance(cfg, dict) or "tolerance" not in cfg:
            raise EvalInfraFailure(f"metric {metric!r} missing 'tolerance'")
        tol = cfg["tolerance"]
        if not isinstance(tol, (int, float)) or isinstance(tol, bool):
            raise EvalInfraFailure(f"metric {metric!r} tolerance must be numeric")
        if isinstance(tol, float) and (math.isnan(tol) or math.isinf(tol)):
            raise EvalInfraFailure(f"metric {metric!r} tolerance must be finite")
    if not isinstance(data["trials"], int) or data["trials"] < 1:
        raise EvalInfraFailure("manifest 'trials' must be an integer >= 1")
    if not isinstance(data["enforce"], bool):
        raise EvalInfraFailure("manifest 'enforce' must be a boolean")
    return data


def _report_path(name: str) -> Path:
    return REPORTS_DIR / f"{name}.json"


def load_report(name: str) -> EvalReport:
    """Load a saved report by name; a missing or malformed one is an infra
    failure (spec: cannot compare → exit 3, never a regression)."""
    path = _report_path(name)
    if not path.exists():
        raise EvalInfraFailure(
            f"cannot compare: report {name!r} not found under {REPORTS_DIR} ({path})"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("missing 'results' list")
        return EvalReport(results=[EvalResult(**r) for r in data["results"]])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise EvalInfraFailure(f"report {name!r} unreadable/malformed: {exc}") from exc


def _check_judge_results(report: EvalReport, name: str) -> None:
    """Judge-reliability lane: any NaN/inf judge value in a report means the
    judge's output was malformed on ≥1 sample — EVAL_INFRA_FAILURE regardless
    of enforce mode (spec I/O matrix: never counted as pass or regression)."""
    for r in report.results:
        for metric in ("accuracy", "faithfulness"):
            v = getattr(r, metric)
            # Malformed judge output = EVAL_INFRA_FAILURE (spec I/O matrix):
            # non-numeric, NaN/inf, booleans, or values outside [0, 1] all
            # mean the judge did not produce a valid bounded score.
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or math.isnan(v)
                or math.isinf(v)
                or not (0.0 <= v <= 1.0)
            ):
                raise EvalInfraFailure(
                    f"judge returned malformed/NaN value for {metric!r} on "
                    f"sample {r.question[:60]!r} in report {name!r}: {v!r}"
                )


def _paired_deltas(a: EvalReport, b: EvalReport, metric: str) -> list[float]:
    base = {r.question: getattr(r, metric) for r in a.results}
    return [
        getattr(rb, metric) - base[rb.question]
        for rb in b.results
        if rb.question in base
    ]


def evaluate_gate(
    baseline: EvalReport, candidate: EvalReport, manifest: dict
) -> list[GateVerdict]:
    """Per-metric paired non-inferiority verdicts, manifest-ordered.

    pass^k (AD-11): the math is implemented and gated on the manifest's
    `trials` config (an open question in SPEC.md — shipped pinned at 1, so
    the extra trials are inert in the first cut). When `trials > 1`, every
    trial must independently be non-inferior: the paired bootstrap is
    re-run under k distinct seeds (deterministic per seed), and a violation
    in ANY trial blocks.
    """
    _check_judge_results(baseline, "baseline")
    _check_judge_results(candidate, "candidate")
    verdicts: list[GateVerdict] = []
    for metric, cfg in manifest["metrics"].items():
        verdicts.append(non_inferiority(baseline, candidate, metric, cfg["tolerance"]))
    if manifest.get("trials", 1) > 1:
        extra: list[GateVerdict] = []
        for trial in range(manifest["trials"]):
            for v in verdicts:
                ci = bootstrap_ci_lower(
                    _paired_deltas(baseline, candidate, v.metric), seed=trial
                )
                if ci < v.tolerance:
                    extra.append(
                        GateVerdict(
                            metric=f"{v.metric}@trial{trial}",
                            tolerance=v.tolerance,
                            mean_delta=v.mean_delta,
                            ci_lower=ci,
                            non_inferior=False,
                        )
                    )
        verdicts.extend(extra)
    return verdicts


def gate_table(verdicts: list[GateVerdict]) -> str:
    rows = [
        "| Metric | Mean Δ | CI95 Lower | Tolerance | Verdict |",
        "|---|---|---|---|---|",
    ]
    for v in verdicts:
        rows.append(
            f"| {v.metric} | {v.mean_delta:+.4f} | {v.ci_lower:+.4f} "
            f"| {v.tolerance:+.4f} | {'non-inferior' if v.non_inferior else 'VIOLATION'} |"
        )
    return "\n".join(rows)


def run_gate(baseline_name: str, candidate_name: str, *, shadow: bool = False) -> int:
    """The gate. Returns the exit code; `shadow=True` forces the advisory
    lane (always exit 0, verdict still printed)."""
    manifest = load_manifest()
    baseline = load_report(baseline_name)
    candidate = load_report(candidate_name)
    verdicts = evaluate_gate(baseline, candidate, manifest)

    blocked = [v for v in verdicts if not v.non_inferior]
    verdict_word = "BLOCKED" if blocked else "PASS"
    enforce = bool(manifest.get("enforce", False)) and not shadow
    mode = "enforce" if enforce else "shadow"
    print(
        f"== eval gate: baseline={baseline_name!r} candidate={candidate_name!r} "
        f"mode={mode} =="
    )
    print(gate_table(verdicts))
    if blocked:
        worst = min(blocked, key=lambda v: v.ci_lower - v.tolerance)
        print(
            f"VERDICT: {verdict_word} — {worst.metric} regressed beyond tolerance "
            f"(CI95 lower {worst.ci_lower:+.4f} < tolerance {worst.tolerance:+.4f})"
        )
    else:
        print(f"VERDICT: {verdict_word} — all pinned metrics non-inferior")

    if not enforce:
        print("shadow mode: advisory only, exit 0")
        return EXIT_PASS
    return EXIT_BLOCKED if blocked else EXIT_PASS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evals.gate",
        description="Paired candidate-vs-baseline non-inferiority eval gate (AD-11 v2)",
    )
    parser.add_argument("baseline", help="baseline report name under evals/reports/")
    parser.add_argument("candidate", help="candidate report name under evals/reports/")
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="advisory mode: print the verdict, always exit 0",
    )
    args = parser.parse_args(argv)
    try:
        return run_gate(args.baseline, args.candidate, shadow=args.shadow)
    except (EvalInfraFailure, JudgeTimeout) as exc:
        # Judge timeouts land in the same lane: a timed-out judge call must
        # never count as a pass or a regression (spec I/O matrix).
        print(f"EVAL_INFRA_FAILURE: {exc}", file=sys.stderr)
        return EXIT_EVAL_INFRA_FAILURE


if __name__ == "__main__":
    sys.exit(main())