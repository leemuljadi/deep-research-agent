from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src import db, ingest


def _mock_connection() -> tuple[MagicMock, MagicMock]:
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = MagicMock()
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection, cursor


class ReplaceDocumentTests(unittest.TestCase):
    def test_replace_document_writes_document_and_chunks_before_one_commit(self) -> None:
        connection, cursor = _mock_connection()
        chunks = [(0, "first", [1.0, 0.0]), (1, "second", [0.0, 1.0])]

        with (
            patch.object(db, "connect", return_value=connection) as connect,
            patch.object(db, "settings", SimpleNamespace(embedding_dim=2)),
        ):
            db.replace_document(
                "doc-1",
                "Title",
                "full content",
                "/corpus/doc.md",
                chunks,
            )

        connect.assert_called_once_with()
        connection.commit.assert_called_once_with()
        calls = cursor.execute.call_args_list
        self.assertEqual(len(calls), 4)
        self.assertIn("INSERT INTO documents", calls[0].args[0])
        self.assertEqual(
            calls[0].args[1],
            ("doc-1", "Title", "/corpus/doc.md", "full content"),
        )
        self.assertIn("DELETE FROM chunks", calls[1].args[0])
        self.assertEqual(calls[1].args[1], ("doc-1",))
        self.assertEqual(calls[2].args[1], ("doc-1", 0, "first", [1.0, 0.0]))
        self.assertEqual(calls[3].args[1], ("doc-1", 1, "second", [0.0, 1.0]))

    def test_replace_document_rejects_dimension_mismatch_before_writes(self) -> None:
        connection, _ = _mock_connection()

        with (
            patch.object(db, "connect", return_value=connection) as connect,
            patch.object(db, "settings", SimpleNamespace(embedding_dim=2)),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"doc_id='doc-1'.*chunk_index=4",
            ):
                db.replace_document(
                    "doc-1",
                    "Title",
                    "content",
                    None,
                    [(0, "valid", [1.0, 0.0]), (4, "bad", [1.0])],
                )

        connect.assert_not_called()

    def test_replace_document_rejects_nan_embedding_before_writes(self) -> None:
        connection, _ = _mock_connection()

        with (
            patch.object(db, "connect", return_value=connection) as connect,
            patch.object(db, "settings", SimpleNamespace(embedding_dim=2)),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"doc_id='doc-1'.*chunk_index=9",
            ):
                db.replace_document(
                    "doc-1",
                    "Title",
                    "content",
                    None,
                    [(9, "bad", [float("nan"), 0.0])],
                )

        connect.assert_not_called()

    def test_replace_document_rejects_none_embedding_before_writes(self) -> None:
        connection, _ = _mock_connection()

        with (
            patch.object(db, "connect", return_value=connection) as connect,
            patch.object(db, "settings", SimpleNamespace(embedding_dim=2)),
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"doc_id='doc-1'.*chunk_index=3",
            ):
                db.replace_document(
                    "doc-1",
                    "Title",
                    "content",
                    None,
                    [(3, "bad", None)],  # type: ignore[list-item]
                )

        connect.assert_not_called()

    def test_replace_document_accepts_empty_chunk_replacement(self) -> None:
        connection, cursor = _mock_connection()

        with (
            patch.object(db, "connect", return_value=connection),
            patch.object(db, "settings", SimpleNamespace(embedding_dim=2)),
        ):
            db.replace_document("doc-1", "Title", "content", None, [])

        connection.commit.assert_called_once_with()
        calls = cursor.execute.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("INSERT INTO documents", calls[0].args[0])
        self.assertIn("DELETE FROM chunks", calls[1].args[0])



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
                patch.object(ingest, "replace_document") as replace,
            ):
                count = ingest.ingest_file(path)

            self.assertEqual(count, 2)
            replace.assert_called_once_with(
                doc_id=ingest._doc_id(path, "document"),
                title="document",
                content="content",
                url=str(path),
                chunks=[
                    (0, "first", [1.0, 0.0]),
                    (1, "second", [0.0, 1.0]),
                ],
            )

    def test_ingest_file_rejects_embedding_count_mismatch_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "chunk_text", return_value=["first", "second"]),
                patch.object(ingest, "embed_texts", return_value=[[1.0]]),
                patch.object(ingest, "replace_document") as replace,
            ):
                with self.assertRaises(ValueError):
                    ingest.ingest_file(path)

            replace.assert_not_called()

    def test_directory_propagates_mismatch_with_path_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document.md"
            path.write_text("content", encoding="utf-8")

            with (
                patch.object(ingest, "init_db"),
                patch.object(ingest, "chunk_text", return_value=["first", "second"]),
                patch.object(ingest, "embed_texts", return_value=[[1.0]]),
                patch.object(ingest, "replace_document"),
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
