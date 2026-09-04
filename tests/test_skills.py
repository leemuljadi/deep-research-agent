from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_FILES = {
    "Claude Code": ROOT / "skills/claude-code/deep-research-agent/SKILL.md",
    "Cursor": ROOT / "skills/cursor/deep-research-agent/SKILL.md",
    "Codex": ROOT / "skills/codex/deep-research-agent/SKILL.md",
}

# Story 11 lands independently. Pin the CAP-8/AD-17 public MCP contract here so
# either channel changing without the other fails this suite at integration.
MCP_JOB_BOUNDARY_VERBS = frozenset(
    {"submit", "poll", "approve", "redirect", "cancel"}
)
JOB_BOUNDARY_ROUTES = {
    "submit": ("POST", "/research"),
    "poll": ("GET", "/runs/{run_id}"),
    "approve": ("POST", "/runs/{run_id}/approve"),
    "redirect": ("POST", "/runs/{run_id}/redirect"),
    "cancel": ("POST", "/runs/{run_id}/cancel"),
}


def _server_routes() -> set[tuple[str, str]]:
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
                and decorator.func.attr in {"get", "post"}
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                continue
            routes.add((decorator.func.attr.upper(), decorator.args[0].value))
    return routes


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing section: {heading}")
    return match.group("body")


def _skill_verbs(text: str) -> set[str]:
    section = _section(text, "Job-boundary verbs")
    return set(re.findall(r"^- `([a-z]+)`\s*$", section, flags=re.MULTILINE))


def _skill_routes(text: str) -> dict[str, tuple[str, str]]:
    section = _section(text, "Supported HTTP contract")
    rows = re.findall(
        r"^\| `([a-z]+)` \| `(GET|POST)` \| `(/[^`]+)` \|\s*$",
        section,
        flags=re.MULTILINE,
    )
    return {verb: (method, route) for verb, method, route in rows}


class AgentSkillsDistributionTests(unittest.TestCase):
    def test_every_harness_skill_and_readme_exists(self) -> None:
        self.assertTrue((ROOT / "skills/README.md").is_file())
        expected = {path.resolve() for path in SKILL_FILES.values()}
        actual = {path.resolve() for path in (ROOT / "skills").rglob("SKILL.md")}
        self.assertEqual(actual, expected)

    def test_server_routes_are_the_job_boundary_contract(self) -> None:
        server_job_routes = {
            (method, route)
            for method, route in _server_routes()
            if route == "/research" or route.startswith("/runs/")
        }
        self.assertEqual(server_job_routes, set(JOB_BOUNDARY_ROUTES.values()))
        self.assertEqual(frozenset(JOB_BOUNDARY_ROUTES), MCP_JOB_BOUNDARY_VERBS)

    def test_each_skill_matches_server_routes_and_mcp_verbs(self) -> None:
        for harness, path in SKILL_FILES.items():
            with self.subTest(harness=harness):
                text = path.read_text(encoding="utf-8")
                self.assertEqual(_skill_verbs(text), MCP_JOB_BOUNDARY_VERBS)
                self.assertEqual(_skill_routes(text), JOB_BOUNDARY_ROUTES)
                self.assertTrue(text.startswith("---\nname: deep-research-agent\n"))
                self.assertRegex(text.split("---", 2)[1], r"\ndescription: .+\n")

    def test_each_skill_documents_cli_and_no_bypass_rule(self) -> None:
        for harness, path in SKILL_FILES.items():
            with self.subTest(harness=harness):
                text = path.read_text(encoding="utf-8")
                self.assertIn("python -m scripts.run_research", text)
                self.assertIn("NEVER bypass the job boundary", text)
                self.assertIn("Do not import or invoke `src.graph`, `src.agents`", text)
                self.assertIn("harness-authored scripts", text)

    def test_readme_documents_each_harness_install_location(self) -> None:
        text = (ROOT / "skills/README.md").read_text(encoding="utf-8")
        for heading, location in (
            ("Claude Code", ".claude/skills/deep-research-agent"),
            ("Cursor", ".cursor/skills/deep-research-agent"),
            ("Codex", ".agents/skills/deep-research-agent"),
        ):
            with self.subTest(harness=heading):
                self.assertIn(f"## {heading}", text)
                self.assertIn(location, text)


if __name__ == "__main__":
    unittest.main()
