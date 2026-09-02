from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from src import ingest


class IngestContractTests(unittest.TestCase):
    def test_ingest_file_writes_ordered_flat_triples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")
            chunks = ["first", "second"]
            embeddings = [[1.0, 0.0], [0.0, 1.0]]

            with (
                patch.object(ingest, "chunk_text", return_value=chunks),
                patch.object(ingest, "embed_texts", return_value=embeddings),
                patch.object(ingest, "upsert_document") as upsert,
                patch.object(ingest, "insert_chunks") as insert,
            ):
                count = ingest.ingest_file(path)

            self.assertEqual(count, 2)
            upsert.assert_called_once()
            doc_id = upsert.call_args.kwargs["doc_id"]
            insert.assert_called_once_with(
                doc_id,
                [(0, "first", [1.0, 0.0]), (1, "second", [0.0, 1.0])],
            )

    def test_ingest_file_rejects_embedding_count_mismatch_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "chunk_text", return_value=["first", "second"]),
                patch.object(ingest, "embed_texts", return_value=[[1.0]]),
                patch.object(ingest, "upsert_document") as upsert,
                patch.object(ingest, "insert_chunks") as insert,
            ):
                with self.assertRaises(ValueError):
                    ingest.ingest_file(path)

            upsert.assert_not_called()
            insert.assert_not_called()

    def test_directory_propagates_mismatch_with_path_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "init_db"),
                patch.object(ingest, "chunk_text", return_value=["first", "second"]),
                patch.object(ingest, "embed_texts", return_value=[[1.0]]),
                patch.object(ingest, "upsert_document"),
                patch.object(ingest, "insert_chunks"),
            ):
                with self.assertRaisesRegex(ValueError, str(path)):
                    ingest.ingest_directory(Path(tmpdir))

    def test_directory_propagates_programming_error_with_path_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "init_db"),
                patch.object(ingest, "ingest_file", side_effect=TypeError("bad contract")),
            ):
                with self.assertRaisesRegex(TypeError, str(path)):
                    ingest.ingest_directory(Path(tmpdir))

    def test_directory_preserves_programming_error_category_for_subclasses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")
            error = json.JSONDecodeError("bad payload", "{}", 0)

            with (
                patch.object(ingest, "init_db"),
                patch.object(ingest, "ingest_file", side_effect=error),
            ):
                with self.assertRaises(ValueError) as raised:
                    ingest.ingest_directory(Path(tmpdir))

            self.assertNotIsInstance(raised.exception, TypeError)
            self.assertIn(str(path), str(raised.exception))

    def test_directory_wraps_unexpected_error_with_path_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "init_db"),
                patch.object(
                    ingest,
                    "ingest_file",
                    side_effect=RuntimeError("database unavailable"),
                ),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    ingest.ingest_directory(Path(tmpdir))

            self.assertEqual(
                str(raised.exception),
                f"Failed to ingest {path}: database unavailable",
            )


    def test_directory_skips_file_operational_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "init_db"),
                patch.object(ingest, "ingest_file", side_effect=OSError("unreadable")),
                redirect_stdout(StringIO()) as output,
            ):
                count = ingest.ingest_directory(Path(tmpdir))

            self.assertEqual(count, 0)
            self.assertIn("[skip] document.md: unreadable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
