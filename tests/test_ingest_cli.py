from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import ingest_corpus


class IngestCorpusCliTests(unittest.TestCase):
    def test_directory_mode_preserves_positional_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    ingest_corpus, "ingest_directory", return_value=4
                ) as ingest_directory,
                patch.object(
                    ingest_corpus, "ingest_github_repository"
                ) as ingest_github,
                patch.object(
                    ingest_corpus, "ingest_youtube_transcript"
                ) as ingest_youtube,
                patch.object(
                    ingest_corpus.sys,
                    "argv",
                    ["scripts.ingest_corpus", tmpdir],
                ),
                redirect_stdout(StringIO()) as output,
            ):
                ingest_corpus.main()

        ingest_directory.assert_called_once_with(Path(tmpdir))
        ingest_github.assert_not_called()
        ingest_youtube.assert_not_called()
        self.assertIn("Indexed 4 chunks", output.getvalue())

    def test_github_mode_dispatches_exactly_once(self) -> None:
        url = "https://github.com/owner/repo"
        with (
            patch.object(
                ingest_corpus, "ingest_github_repository", return_value=7
            ) as ingest_github,
            patch.object(ingest_corpus, "ingest_directory") as ingest_directory,
            patch.object(
                ingest_corpus, "ingest_youtube_transcript"
            ) as ingest_youtube,
            patch.object(
                ingest_corpus.sys,
                "argv",
                ["scripts.ingest_corpus", "--github", url],
            ),
            redirect_stdout(StringIO()) as output,
        ):
            ingest_corpus.main()

        ingest_github.assert_called_once_with(url)
        ingest_directory.assert_not_called()
        ingest_youtube.assert_not_called()
        self.assertIn("Indexed 7 chunks", output.getvalue())
        self.assertIn(url, output.getvalue())

    def test_youtube_mode_dispatches_exactly_once(self) -> None:
        video_id = "dQw4w9WgXcQ"
        with (
            patch.object(
                ingest_corpus, "ingest_youtube_transcript", return_value=5
            ) as ingest_youtube,
            patch.object(ingest_corpus, "ingest_directory") as ingest_directory,
            patch.object(
                ingest_corpus, "ingest_github_repository"
            ) as ingest_github,
            patch.object(
                ingest_corpus.sys,
                "argv",
                ["scripts.ingest_corpus", "--youtube", video_id],
            ),
            redirect_stdout(StringIO()) as output,
        ):
            ingest_corpus.main()

        ingest_youtube.assert_called_once_with(video_id)
        ingest_directory.assert_not_called()
        ingest_github.assert_not_called()
        self.assertIn("Indexed 5 chunks", output.getvalue())
        self.assertIn(video_id, output.getvalue())

    def test_missing_flag_value_prints_usage_before_io(self) -> None:
        with (
            patch.object(ingest_corpus, "ingest_directory") as ingest_directory,
            patch.object(
                ingest_corpus, "ingest_github_repository"
            ) as ingest_github,
            patch.object(
                ingest_corpus, "ingest_youtube_transcript"
            ) as ingest_youtube,
            patch.object(
                ingest_corpus.sys,
                "argv",
                ["scripts.ingest_corpus", "--github"],
            ),
            redirect_stderr(StringIO()) as error,
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                ingest_corpus.main()

        ingest_directory.assert_not_called()
        ingest_github.assert_not_called()
        ingest_youtube.assert_not_called()
        self.assertIn("Usage:", error.getvalue())

    def test_unknown_flag_prints_usage_before_io(self) -> None:
        with (
            patch.object(ingest_corpus, "ingest_directory") as ingest_directory,
            patch.object(
                ingest_corpus, "ingest_github_repository"
            ) as ingest_github,
            patch.object(
                ingest_corpus, "ingest_youtube_transcript"
            ) as ingest_youtube,
            patch.object(
                ingest_corpus.sys,
                "argv",
                ["scripts.ingest_corpus", "--unknown", "value"],
            ),
            redirect_stderr(StringIO()) as error,
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                ingest_corpus.main()

        ingest_directory.assert_not_called()
        ingest_github.assert_not_called()
        ingest_youtube.assert_not_called()
        self.assertIn("Usage:", error.getvalue())


if __name__ == "__main__":
    unittest.main()
