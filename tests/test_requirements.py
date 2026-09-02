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


if __name__ == "__main__":
    unittest.main()
