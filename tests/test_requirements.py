from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RequirementListTests(unittest.TestCase):
    def test_requirements_are_unique_and_exclude_stale_direct_dependencies(self) -> None:
        requirements = [
            line.strip()
            for line in (PROJECT_ROOT / "requirements.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        names = []
        for requirement in requirements:
            match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
            self.assertIsNotNone(match, requirement)
            names.append(re.sub(r"[-_.]+", "-", match.group(0).lower()))

        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("langchain-core", names)
        self.assertNotIn("numpy", names)
        self.assertEqual(names.count("opentelemetry-instrumentation-requests"), 1)

    def test_dependency_floors_encode_supply_chain_bans(self) -> None:
        requirements = {}
        for line in (
            PROJECT_ROOT / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*", stripped)
            self.assertIsNotNone(
                match,
                f"unparseable requirement line: {stripped!r}",
            )
            name = re.sub(r"[-_.]+", "-", match.group(0).lower())
            requirements[name] = stripped
        self.assertEqual(len(requirements), len(set(requirements)))

        # AD-12: never install litellm 1.82.7/1.82.8 (malicious PyPI uploads,
        # 2026-03) — the range itself must exclude them.
        self.assertRequirementFloor(requirements, "litellm", (1, 82, 9))
        # AD-12: the langgraph floor must not span the 1.0 breaking change.
        self.assertRequirementFloor(requirements, "langgraph", (1, 0))
        # YouTube has no stable public transcript endpoint; the ingest-only
        # adapter uses the current fetch API while keeping imports lazy.
        self.assertRequirementFloor(
            requirements, "youtube-transcript-api", (1, 2, 4)
        )
        # AD-17: server-side MCP must stay on the v2 protocol line and
        # standalone FastMCP must stay on its v4 application-framework line.
        self.assertRequirementFloor(requirements, "mcp", (2,))
        self.assertRequirementFloor(requirements, "fastmcp", (4,))

    def assertRequirementFloor(
        self, requirements: dict[str, str], name: str, minimum: tuple[int, ...]
    ) -> None:
        self.assertIn(name, requirements, f"{name} missing from requirements.txt")
        requirement = requirements[name]
        match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", requirement)
        self.assertIsNotNone(
            match,
            f"{requirement!r}: expected a '>=' minimum-version floor",
        )
        floor = tuple(int(part) for part in match.group(1).split("."))
        self.assertGreaterEqual(
            len(floor),
            len(minimum),
            f"{requirement!r}: floor must state at least "
            f"{'.'.join(str(part) for part in minimum)}",
        )
        floor = floor[: len(minimum)]
        self.assertGreaterEqual(
            floor,
            minimum,
            f"{name} floor is stale: {requirement!r} admits versions below "
            f"{'.'.join(str(part) for part in minimum)}; "
            f"required minimum is {name}>={'.'.join(str(part) for part in minimum)}",
        )


if __name__ == "__main__":
    unittest.main()
