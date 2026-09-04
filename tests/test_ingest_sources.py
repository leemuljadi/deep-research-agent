from __future__ import annotations

import hashlib
import io
import tarfile
import unittest
from email.message import Message
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import patch

from src import ingest, ingest_sources


COMMIT_SHA = "a" * 40
VIDEO_ID = "dQw4w9WgXcQ"


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative_path, content in files.items():
            info = tarfile.TarInfo(f"owner-repo-{COMMIT_SHA}/{relative_path}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class SourceDocumentContractTests(unittest.TestCase):
    def test_namespaced_document_id_uses_sha1_16(self) -> None:
        identity = f"owner/repo@{COMMIT_SHA}"

        self.assertEqual(
            ingest.source_doc_id("github", identity),
            hashlib.sha1(f"github:{identity}".encode()).hexdigest()[:16],
        )
        self.assertEqual(
            ingest.source_doc_id("youtube", VIDEO_ID),
            hashlib.sha1(f"youtube:{VIDEO_ID}".encode()).hexdigest()[:16],
        )

    def test_text_document_emits_ordered_replace_document_triples(self) -> None:
        chunks = ["first", "second"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        with (
            patch.object(ingest, "chunk_text", return_value=chunks),
            patch.object(ingest, "embed_texts", return_value=embeddings),
            patch.object(ingest, "replace_document") as replace,
        ):
            count = ingest.ingest_text_document(
                doc_id="source-id",
                title="Source title",
                content="source content",
                url="https://example.test/source",
            )

        self.assertEqual(count, 2)
        replace.assert_called_once_with(
            doc_id="source-id",
            title="Source title",
            content="source content",
            url="https://example.test/source",
            chunks=[
                (0, "first", [1.0, 0.0]),
                (1, "second", [0.0, 1.0]),
            ],
        )

    def test_text_document_rejects_embedding_count_mismatch_before_write(self) -> None:
        with (
            patch.object(ingest, "chunk_text", return_value=["first", "second"]),
            patch.object(ingest, "embed_texts", return_value=[[1.0]]),
            patch.object(ingest, "replace_document") as replace,
        ):
            with self.assertRaises(ValueError):
                ingest.ingest_text_document(
                    doc_id="source-id",
                    title="Source title",
                    content="source content",
                    url=None,
                )

        replace.assert_not_called()


class GitHubAdapterTests(unittest.TestCase):
    def test_load_repository_normalizes_readme_then_sorted_code(self) -> None:
        archive = _archive(
            {
                "src/z.py": b"print('z')\n",
                "README.md": b"# Demo\n",
                "src/a.py": b"print('a')\n",
                "asset.png": b"not text",
                "src/binary.py": b"bad\x00data",
            }
        )
        with (
            patch.object(
                ingest_sources, "_fetch_github_commit", return_value=COMMIT_SHA
            ) as fetch_commit,
            patch.object(
                ingest_sources, "_fetch_github_archive", return_value=archive
            ) as fetch_archive,
        ):
            document = ingest_sources.load_github_repository(
                "https://github.com/Owner/Repo.git/"
            )

        fetch_commit.assert_called_once_with("owner", "repo")
        fetch_archive.assert_called_once_with("owner", "repo", COMMIT_SHA)
        self.assertEqual(
            document.doc_id,
            ingest.source_doc_id("github", f"owner/repo@{COMMIT_SHA}"),
        )
        self.assertEqual(document.title, "GitHub owner/repo @ aaaaaaaaaaaa")
        self.assertEqual(
            document.url,
            f"https://github.com/owner/repo/tree/{COMMIT_SHA}",
        )
        self.assertIn("Commit: " + COMMIT_SHA, document.content)
        readme = document.content.index("## File: README.md")
        source_a = document.content.index("## File: src/a.py")
        source_z = document.content.index("## File: src/z.py")
        self.assertLess(readme, source_a)
        self.assertLess(source_a, source_z)
        self.assertNotIn("asset.png", document.content)
        self.assertNotIn("binary.py", document.content)

    def test_root_readme_precedes_nested_readmes(self) -> None:
        archive = _archive(
            {
                "docs/README.md": b"# Nested\n",
                "README.md": b"# Root\n",
            }
        )
        with (
            patch.object(
                ingest_sources, "_fetch_github_commit", return_value=COMMIT_SHA
            ),
            patch.object(
                ingest_sources, "_fetch_github_archive", return_value=archive
            ),
        ):
            document = ingest_sources.load_github_repository(
                "https://github.com/owner/repo"
            )

        self.assertLess(
            document.content.index("## File: README.md"),
            document.content.index("## File: docs/README.md"),
        )

    def test_repository_rejects_invalid_url_before_network(self) -> None:
        with patch.object(ingest_sources, "_fetch_github_commit") as fetch_commit:
            with self.assertRaisesRegex(ValueError, "GitHub repository URL"):
                ingest_sources.load_github_repository(
                    "https://gitlab.com/owner/repo"
                )

        fetch_commit.assert_not_called()

    def test_repository_403_rate_limit_is_source_error_without_secret(self) -> None:
        headers = Message()
        headers["X-RateLimit-Remaining"] = "0"
        error = HTTPError(
            url="https://api.github.com/repos/owner/repo/commits/HEAD",
            code=403,
            msg="Forbidden",
            hdrs=headers,
            fp=io.BytesIO(b'{"message":"API rate limit exceeded"}'),
        )
        with (
            patch.object(ingest_sources, "_urlopen", side_effect=error),
            patch.dict("os.environ", {"GITHUB_TOKEN": "do-not-leak"}),
        ):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError, "rate limit"
            ) as raised:
                ingest_sources.load_github_repository(
                    "https://github.com/owner/repo"
                )

        self.assertNotIn("do-not-leak", str(raised.exception))

    def test_repository_404_is_source_error(self) -> None:
        error = HTTPError(
            url="https://api.github.com/repos/owner/missing/commits/HEAD",
            code=404,
            msg="Not Found",
            hdrs=Message(),
            fp=io.BytesIO(b'{"message":"Not Found"}'),
        )
        with patch.object(ingest_sources, "_urlopen", side_effect=error):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError, "not found or inaccessible"
            ):
                ingest_sources.load_github_repository(
                    "https://github.com/owner/missing"
                )

    def test_repository_with_no_supported_text_is_rejected(self) -> None:
        archive = _archive({"asset.png": b"binary-only"})
        with (
            patch.object(
                ingest_sources, "_fetch_github_commit", return_value=COMMIT_SHA
            ),
            patch.object(
                ingest_sources, "_fetch_github_archive", return_value=archive
            ),
        ):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError, "no supported text files"
            ):
                ingest_sources.load_github_repository(
                    "https://github.com/owner/repo"
                )

    def test_repository_rejects_oversized_archive(self) -> None:
        with (
            patch.object(ingest_sources, "GITHUB_MAX_ARCHIVE_BYTES", 10),
            patch.object(
                ingest_sources, "_fetch_github_commit", return_value=COMMIT_SHA
            ),
            patch.object(
                ingest_sources, "_fetch_github_archive", return_value=b"x" * 11
            ),
        ):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError, "archive exceeds"
            ):
                ingest_sources.load_github_repository(
                    "https://github.com/owner/repo"
                )

    def test_repository_rejects_excessive_expanded_size(self) -> None:
        archive = _archive({"large.bin": b"x" * 101})
        with (
            patch.object(ingest_sources, "GITHUB_MAX_EXPANDED_BYTES", 100),
            patch.object(
                ingest_sources, "_fetch_github_commit", return_value=COMMIT_SHA
            ),
            patch.object(
                ingest_sources, "_fetch_github_archive", return_value=archive
            ),
        ):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError, "expanded archive exceeds"
            ):
                ingest_sources.load_github_repository(
                    "https://github.com/owner/repo"
                )

    def test_repository_content_truncation_is_explicit_and_bounded(self) -> None:
        archive = _archive({"README.md": b"x" * 500})
        with (
            patch.object(ingest_sources, "GITHUB_MAX_CONTENT_CHARS", 160),
            patch.object(
                ingest_sources, "_fetch_github_commit", return_value=COMMIT_SHA
            ),
            patch.object(
                ingest_sources, "_fetch_github_archive", return_value=archive
            ),
        ):
            document = ingest_sources.load_github_repository(
                "https://github.com/owner/repo"
            )

        self.assertLessEqual(len(document.content), 160)
        self.assertIn("[truncated", document.content)

    def test_ingest_repository_initializes_then_uses_shared_contract(self) -> None:
        document = ingest_sources.SourceDocument(
            doc_id="github-id",
            title="GitHub owner/repo",
            content="repository text",
            url="https://github.com/owner/repo/tree/sha",
        )
        with (
            patch.object(
                ingest_sources, "load_github_repository", return_value=document
            ) as load,
            patch.object(ingest_sources, "init_db") as init_db,
            patch.object(
                ingest_sources, "ingest_text_document", return_value=3
            ) as ingest_document,
        ):
            count = ingest_sources.ingest_github_repository(
                "https://github.com/owner/repo"
            )

        self.assertEqual(count, 3)
        load.assert_called_once_with("https://github.com/owner/repo")
        init_db.assert_called_once_with()
        ingest_document.assert_called_once_with(
            doc_id=document.doc_id,
            title=document.title,
            content=document.content,
            url=document.url,
        )

    def test_repository_fetch_failure_never_initializes_or_writes(self) -> None:
        with (
            patch.object(
                ingest_sources,
                "load_github_repository",
                side_effect=ingest_sources.SourceIngestError("empty repository"),
            ),
            patch.object(ingest_sources, "init_db") as init_db,
            patch.object(ingest_sources, "ingest_text_document") as ingest_document,
        ):
            with self.assertRaises(ingest_sources.SourceIngestError):
                ingest_sources.ingest_github_repository(
                    "https://github.com/owner/repo"
                )

        init_db.assert_not_called()
        ingest_document.assert_not_called()


class YouTubeAdapterTests(unittest.TestCase):
    def test_load_transcript_normalizes_snippet_text(self) -> None:
        snippets = [
            SimpleNamespace(text="  Hello   &amp; goodbye  "),
            SimpleNamespace(text="Line\n two"),
            SimpleNamespace(text="   "),
        ]
        with patch.object(
            ingest_sources, "_fetch_youtube_transcript", return_value=snippets
        ) as fetch:
            document = ingest_sources.load_youtube_transcript(VIDEO_ID)

        fetch.assert_called_once_with(VIDEO_ID)
        self.assertEqual(
            document.doc_id,
            ingest.source_doc_id("youtube", VIDEO_ID),
        )
        self.assertEqual(document.title, f"YouTube transcript: {VIDEO_ID}")
        self.assertEqual(
            document.url,
            f"https://www.youtube.com/watch?v={VIDEO_ID}",
        )
        self.assertEqual(document.content, "Hello & goodbye\nLine two")

    def test_transcript_rejects_invalid_video_id_before_network(self) -> None:
        with patch.object(ingest_sources, "_fetch_youtube_transcript") as fetch:
            with self.assertRaisesRegex(ValueError, "video ID"):
                ingest_sources.load_youtube_transcript("not-a-video-url")

        fetch.assert_not_called()

    def test_empty_transcript_is_rejected(self) -> None:
        with patch.object(
            ingest_sources,
            "_fetch_youtube_transcript",
            return_value=[SimpleNamespace(text="   ")],
        ):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError, "empty transcript"
            ):
                ingest_sources.load_youtube_transcript(VIDEO_ID)

    def test_provider_failure_is_wrapped_as_source_error(self) -> None:
        class FailingYouTubeTranscriptApi:
            def fetch(self, video_id: str) -> None:
                raise RuntimeError(f"captions disabled for {video_id}")

        module = ModuleType("youtube_transcript_api")
        module.YouTubeTranscriptApi = FailingYouTubeTranscriptApi
        with patch.dict("sys.modules", {"youtube_transcript_api": module}):
            with self.assertRaisesRegex(
                ingest_sources.SourceIngestError,
                f"transcript unavailable for {VIDEO_ID}",
            ) as raised:
                ingest_sources._fetch_youtube_transcript(VIDEO_ID)

        self.assertIn("captions disabled", str(raised.exception))

    def test_transcript_content_truncation_is_explicit_and_bounded(self) -> None:
        with (
            patch.object(ingest_sources, "YOUTUBE_MAX_CONTENT_CHARS", 80),
            patch.object(
                ingest_sources,
                "_fetch_youtube_transcript",
                return_value=[SimpleNamespace(text="word " * 100)],
            ),
        ):
            document = ingest_sources.load_youtube_transcript(VIDEO_ID)

        self.assertLessEqual(len(document.content), 80)
        self.assertIn("[truncated", document.content)

    def test_unavailable_transcript_never_initializes_or_writes(self) -> None:
        with (
            patch.object(
                ingest_sources,
                "load_youtube_transcript",
                side_effect=ingest_sources.SourceIngestError(
                    "transcript unavailable"
                ),
            ),
            patch.object(ingest_sources, "init_db") as init_db,
            patch.object(ingest_sources, "ingest_text_document") as ingest_document,
        ):
            with self.assertRaises(ingest_sources.SourceIngestError):
                ingest_sources.ingest_youtube_transcript(VIDEO_ID)

        init_db.assert_not_called()
        ingest_document.assert_not_called()

    def test_ingest_transcript_initializes_then_uses_shared_contract(self) -> None:
        document = ingest_sources.SourceDocument(
            doc_id="youtube-id",
            title=f"YouTube transcript: {VIDEO_ID}",
            content="transcript text",
            url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
        )
        with (
            patch.object(
                ingest_sources, "load_youtube_transcript", return_value=document
            ),
            patch.object(ingest_sources, "init_db") as init_db,
            patch.object(
                ingest_sources, "ingest_text_document", return_value=2
            ) as ingest_document,
        ):
            count = ingest_sources.ingest_youtube_transcript(VIDEO_ID)

        self.assertEqual(count, 2)
        init_db.assert_called_once_with()
        ingest_document.assert_called_once_with(
            doc_id=document.doc_id,
            title=document.title,
            content=document.content,
            url=document.url,
        )


if __name__ == "__main__":
    unittest.main()
