from __future__ import annotations

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import evals.eval_harness as h
from evals import gate
from evals.eval_harness import EvalReport, EvalResult, bootstrap_ci_lower, non_inferiority


def _report(*rows: tuple[str, float, float]) -> EvalReport:
    """Report from (question, accuracy, faithfulness) rows."""
    return EvalReport(
        results=[
            EvalResult(
                question=q,
                accuracy=a,
                faithfulness=f,
                cost_usd=0.0,
                latency_s=1.0,
                tokens=10,
            )
            for q, a, f in rows
        ]
    )


MANIFEST = {
    "dataset": "golden_set",
    "judge": {"prompt_version": "v1"},
    "decoding": {"temperature": 0.0},
    "metrics": {
        "faithfulness": {"tolerance": -0.05},
        "accuracy": {"tolerance": -0.05},
    },
    "trials": 1,
    "max_samples": None,
    "enforce": False,
    "timeout_s": 60,
}


class BootstrapCITests(unittest.TestCase):
    def test_deterministic_for_same_inputs(self) -> None:
        deltas = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02, 0.01, -0.03]
        self.assertEqual(
            bootstrap_ci_lower(deltas, n_boot=500, seed=7),
            bootstrap_ci_lower(deltas, n_boot=500, seed=7),
        )

    def test_identical_deltas_bound_equals_delta(self) -> None:
        # All resamples of an identical delta set have the same mean.
        self.assertAlmostEqual(bootstrap_ci_lower([0.1] * 6), 0.1)

    def test_tight_positive_deltas_stay_above_zero(self) -> None:
        deltas = [0.02, 0.03, 0.02, 0.01, 0.03, 0.02]
        self.assertGreater(bootstrap_ci_lower(deltas), 0.0)

    def test_empty_deltas_gives_negative_infinity(self) -> None:
        self.assertEqual(bootstrap_ci_lower([]), -math.inf)

    def test_wide_noise_bound_below_mean(self) -> None:
        deltas = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        self.assertLess(bootstrap_ci_lower(deltas), 0.0)


class NonInferiorityTests(unittest.TestCase):
    def test_candidate_equal_to_baseline_passes(self) -> None:
        base = _report(("q1", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.9))
        v = non_inferiority(base, cand, "faithfulness", -0.05)
        self.assertTrue(v.non_inferior)
        self.assertAlmostEqual(v.mean_delta, 0.0)
        self.assertAlmostEqual(v.ci_lower, 0.0)

    def test_big_regression_blocks(self) -> None:
        base = _report(("q1", 0.8, 0.9), ("q2", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.85), ("q2", 0.8, 0.85))
        v = non_inferiority(base, cand, "faithfulness", -0.05)
        self.assertFalse(v.non_inferior)
        self.assertLess(v.ci_lower, -0.05)

    def test_small_regression_within_tolerance_passes(self) -> None:
        base = _report(("q1", 0.8, 0.9), ("q2", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.88), ("q2", 0.8, 0.88))
        v = non_inferiority(base, cand, "faithfulness", -0.05)
        self.assertTrue(v.non_inferior)

    def test_unpaired_samples_are_ignored(self) -> None:
        base = _report(("q1", 0.8, 0.9), ("extra-base", 0.5, 0.5))
        cand = _report(("q1", 0.8, 0.9), ("extra-cand", 0.99, 0.99))
        v = non_inferiority(base, cand, "faithfulness", -0.05)
        self.assertTrue(v.non_inferior)
        self.assertAlmostEqual(v.mean_delta, 0.0)

    def test_improvement_passes(self) -> None:
        base = _report(("q1", 0.8, 0.9))
        cand = _report(("q1", 0.9, 0.95))
        v = non_inferiority(base, cand, "faithfulness", -0.05)
        self.assertTrue(v.non_inferior)


class ManifestTests(unittest.TestCase):
    def test_repo_manifest_parses_and_pins_enforce_false(self) -> None:
        m = gate.load_manifest()
        self.assertEqual(m["dataset"], "golden_set")
        self.assertFalse(m["enforce"])  # shadow default until instrumented
        self.assertGreaterEqual(m["trials"], 1)
        for metric, cfg in m["metrics"].items():
            self.assertIn("tolerance", cfg)

    def _write_manifest(self, tmp: Path, data: dict) -> Path:
        import yaml

        path = tmp / "manifest.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def test_missing_required_key_is_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bad = {k: v for k, v in MANIFEST.items() if k != "enforce"}
            with self.assertRaises(gate.EvalInfraFailure):
                gate.load_manifest(self._write_manifest(tmp, bad))

    def test_non_boolean_enforce_is_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bad = dict(MANIFEST, enforce="yes")
            with self.assertRaises(gate.EvalInfraFailure):
                gate.load_manifest(self._write_manifest(tmp, bad))

    def test_bad_trials_is_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bad = dict(MANIFEST, trials=0)
            with self.assertRaises(gate.EvalInfraFailure):
                gate.load_manifest(self._write_manifest(tmp, bad))

    def test_metric_missing_tolerance_is_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bad = dict(MANIFEST, metrics={"faithfulness": {}})
            with self.assertRaises(gate.EvalInfraFailure):
                gate.load_manifest(self._write_manifest(tmp, bad))

    def test_unreadable_manifest_is_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.yaml"
            path.write_text("{ not: [valid yaml")
            with self.assertRaises(gate.EvalInfraFailure):
                gate.load_manifest(path)


class GateLanesTests(unittest.TestCase):
    """Exit-code lanes driven through run_gate with patched IO."""

    def _patch_manifest_and_reports(
        self,
        baseline: EvalReport | None,
        candidate: EvalReport | None,
        manifest: dict | None = None,
    ):
        manifest = manifest or MANIFEST
        import contextlib

        @contextlib.contextmanager
        def _patches():
            with (
                patch.object(gate, "load_manifest", return_value=manifest),
                patch.object(gate, "load_report") as lr,
            ):
                if baseline is None:
                    lr.side_effect = gate.EvalInfraFailure("missing report")
                else:
                    lr.side_effect = [baseline, candidate]
                yield

        return _patches()

    def test_within_tolerance_enforce_exits_zero(self) -> None:
        base = _report(("q1", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.9))
        enforce = dict(MANIFEST, enforce=True)
        with self._patch_manifest_and_reports(base, cand, enforce):
            self.assertEqual(gate.run_gate("b", "c"), 0)
        base = _report(("q1", 0.8, 0.9), ("q2", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.5), ("q2", 0.8, 0.5))
        enforce = dict(MANIFEST, enforce=True)
        with self._patch_manifest_and_reports(base, cand, enforce):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = gate.run_gate("b", "c")
        self.assertEqual(code, 1)
        self.assertIn("faithfulness", out.getvalue())
        self.assertIn("BLOCKED", out.getvalue())

    def test_shadow_always_exits_zero_even_when_blocked(self) -> None:
        base = _report(("q1", 0.8, 0.9), ("q2", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.5), ("q2", 0.8, 0.5))
        enforce = dict(MANIFEST, enforce=True)
        with self._patch_manifest_and_reports(base, cand, enforce):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = gate.run_gate("b", "c", shadow=True)
        self.assertEqual(code, 0)  # advisory despite BLOCKED
        self.assertIn("BLOCKED", out.getvalue())

    def test_shadow_manifest_reports_but_exits_zero(self) -> None:
        base = _report(("q1", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.5))
        with self._patch_manifest_and_reports(base, cand, dict(MANIFEST)):
            self.assertEqual(gate.run_gate("b", "c"), 0)

    def test_nan_judge_value_is_infra_failure(self) -> None:
        base = _report(("q1", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.9))
        # Corrupt the candidate report post-construction, as a malformed
        # judge JSON round-trip would (NaN survives json round-trips).
        cand.results[0].faithfulness = float("nan")
        enforce = dict(MANIFEST, enforce=True)
        with self._patch_manifest_and_reports(base, cand, enforce):
            with self.assertRaises(gate.EvalInfraFailure):
                gate.run_gate("b", "c")

    def test_missing_report_is_infra_failure(self) -> None:
        with self._patch_manifest_and_reports(None, None):
            with self.assertRaises(gate.EvalInfraFailure):
                gate.run_gate("b", "c")


class ReportLoadingTests(unittest.TestCase):
    def test_malformed_report_json_is_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            reports = tmp / "reports"
            reports.mkdir()
            (reports / "bad.json").write_text("{results: [oops")
            with patch.object(gate, "REPORTS_DIR", reports):
                with self.assertRaises(gate.EvalInfraFailure):
                    gate.load_report("bad")

    def test_missing_report_file_is_infra_failure_with_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td) / "reports"
            reports.mkdir()
            with patch.object(gate, "REPORTS_DIR", reports):
                with self.assertRaises(gate.EvalInfraFailure) as cm:
                    gate.load_report("nope")
            self.assertIn("cannot compare", str(cm.exception))

    def test_report_round_trip_loads_results(self) -> None:
        report = _report(("q1", 0.8, 0.9))
        with tempfile.TemporaryDirectory() as td:
            reports = Path(td) / "reports"
            reports.mkdir()
            path = reports / "good.json"
            path.write_text(json.dumps(report.to_dict()))
            with patch.object(gate, "REPORTS_DIR", reports):
                loaded = gate.load_report("good")
        self.assertEqual(len(loaded.results), 1)
        self.assertAlmostEqual(loaded.results[0].faithfulness, 0.9)


class PassKTests(unittest.TestCase):
    def test_trials_greater_than_one_blocks_on_any_trial_violation(self) -> None:
        # Candidate is worse on faithfulness: with several trials the paired
        # bootstrap must flag a violation in at least one trial's resample.
        base = _report(("q1", 0.8, 0.9), ("q2", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.4), ("q2", 0.8, 0.4))
        m = dict(MANIFEST, trials=3)
        verdicts = gate.evaluate_gate(base, cand, m)
        trials = [v for v in verdicts if "@trial" in v.metric]
        self.assertTrue(trials)
        self.assertTrue(any(not v.non_inferior for v in trials))

    def test_trials_one_is_inert(self) -> None:
        base = _report(("q1", 0.8, 0.9))
        cand = _report(("q1", 0.8, 0.4))
        verdicts = gate.evaluate_gate(base, cand, dict(MANIFEST, trials=1))
        self.assertTrue(all("@trial" not in v.metric for v in verdicts))
        self.assertFalse(verdicts[0].non_inferior)


class GateCliTests(unittest.TestCase):
    """End-to-end CLI lanes: subprocess against the module entrypoint."""

    def _run(self, *args: str, reports: dict[str, str] | None = None,
             manifest: str | None = None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in __import__("os").environ.items()}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            if reports:
                rdir = tmp / "reports"
                rdir.mkdir()
                for name, rows in reports.items():
                    (rdir / f"{name}.json").write_text(rows)
            if manifest:
                (tmp / "manifest.yaml").write_text(manifest)
            # The wrapper patches `load_manifest` directly (its default arg
            # binds MANIFEST_PATH at def time, so patching the constant is
            # inert) and REPORTS_DIR, which load_report reads at call time.
            wrapper = tmp / "run_gate.py"
            wrapper.write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
                "from unittest.mock import patch\n"
                "from evals import gate\n"
                f"gate.REPORTS_DIR = __import__('pathlib').Path({str(tmp / 'reports')!r})\n"
                f"_m = gate.load_manifest(__import__('pathlib').Path({str(tmp / 'manifest.yaml')!r}))\n"
                "with patch.object(gate, 'load_manifest', return_value=_m):\n"
                "    sys.exit(gate.main(sys.argv[1:]))\n"
            )
            return subprocess.run(
                [sys.executable, str(wrapper), *args],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )

    _GOOD_MANIFEST = (
        "dataset: golden_set\n"
        "judge: {prompt_version: v1}\n"
        "metrics:\n"
        "  faithfulness: {tolerance: -0.05}\n"
        "trials: 1\n"
        "enforce: true\n"
    )

    def test_cli_enforce_block_exits_one(self) -> None:
        base = json.dumps(
            _report(("q1", 0.8, 0.9)).to_dict()
        )
        cand = json.dumps(
            _report(("q1", 0.8, 0.5)).to_dict()
        )
        proc = self._run(
            "base", "cand", reports={"base": base, "cand": cand},
            manifest=self._GOOD_MANIFEST,
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("BLOCKED", proc.stdout)
        self.assertIn("faithfulness", proc.stdout)

    def test_cli_shadow_blocked_input_exits_zero(self) -> None:
        base = json.dumps(_report(("q1", 0.8, 0.9)).to_dict())
        cand = json.dumps(_report(("q1", 0.8, 0.5)).to_dict())
        proc = self._run(
            "--shadow", "base", "cand",
            reports={"base": base, "cand": cand},
            manifest=self._GOOD_MANIFEST,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("BLOCKED", proc.stdout)

    def test_cli_missing_baseline_exits_three(self) -> None:
        cand = json.dumps(_report(("q1", 0.8, 0.9)).to_dict())
        proc = self._run(
            "nope", "cand", reports={"cand": cand},
            manifest=self._GOOD_MANIFEST,
        )
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("EVAL_INFRA_FAILURE", proc.stderr)

    def test_cli_nan_judge_exits_three_even_in_shadow(self) -> None:
        base = json.dumps(_report(("q1", 0.8, 0.9)).to_dict())
        bad = _report(("q1", 0.8, 0.9))
        bad.results[0].faithfulness = float("nan")
        proc = self._run(
            "--shadow", "base", "cand",
            reports={"base": base, "cand": json.dumps(bad.to_dict())},
            manifest=self._GOOD_MANIFEST,
        )
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)

    def test_cli_manifest_enforce_toggle_controls_exit(self) -> None:
        base = json.dumps(_report(("q1", 0.8, 0.9)).to_dict())
        cand = json.dumps(_report(("q1", 0.8, 0.9)).to_dict())
        shadow_manifest = self._GOOD_MANIFEST.replace("enforce: true", "enforce: false")
        proc = self._run(
            "base", "cand", reports={"base": base, "cand": cand},
            manifest=shadow_manifest,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("shadow mode", proc.stdout)


class JudgeTimeoutTests(unittest.TestCase):
    """The harness raises JudgeTimeout on litellm.Timeout instead of the
    silent 0.5/lexical fallback (AD-11: a timed-out judge is infra failure),
    and the gate maps that to the exit-3 lane."""

    def test_faithfulness_timeout_propagates(self) -> None:
        import litellm
        from src.schemas import ResearchReport, Source

        report = ResearchReport(
            summary="s", findings=["f"],
            sources=[Source(title="t", url="u", snippet="s")], confidence=0.9,
        )
        with patch.object(
            h, "chat", side_effect=litellm.Timeout("slow", model="chat", llm_provider="x")
        ):
            with self.assertRaises(h.JudgeTimeout):
                h._faithfulness(report)

    def test_accuracy_timeout_propagates(self) -> None:
        import litellm
        from src.schemas import ResearchReport

        report = ResearchReport(summary="s", findings=[], sources=[], confidence=0.9)
        with patch.object(
            h, "chat", side_effect=litellm.Timeout("slow", model="chat", llm_provider="x")
        ):
            with self.assertRaises(h.JudgeTimeout):
                h._accuracy(report, h.EvalSample(question="q", expected_keywords=["k"]))

    def test_generic_failure_still_falls_back(self) -> None:
        # Non-timeout judge failures keep the pre-existing fallback behavior.
        from src.schemas import ResearchReport, Source

        report = ResearchReport(
            summary="s", findings=["f"],
            sources=[Source(title="t", url="u", snippet="s")], confidence=0.9,
        )
        with patch.object(h, "chat", side_effect=RuntimeError("blip")):
            self.assertEqual(h._faithfulness(report), 0.5)

    def test_timeout_env_guarded_parsing(self) -> None:
        # Guarded env parsing: unset/empty/garbage/non-finite/<=0 → 60.
        cases = [
            (None, 60.0),
            ("", 60.0),
            ("abc", 60.0),
            ("0", 60.0),
            ("-5", 60.0),
            ("inf", 60.0),
            ("nan", 60.0),
            ("30", 30.0),
            ("2.5", 2.5),
        ]
        for raw, expected in cases:
            env = {"EVAL_JUDGE_TIMEOUT_S": raw} if raw is not None else {}
            with patch.dict("os.environ", env):
                self.assertEqual(h._judge_timeout_s(), expected, f"raw={raw!r}")

    def test_gate_maps_timeout_to_exit_three(self) -> None:
        with patch.object(
            gate, "run_gate", side_effect=h.JudgeTimeout("judge timed out")
        ):
            rc = gate.main(["b", "c"])
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()