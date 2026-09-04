"""Ingest-time adapters for bounded GitHub snapshots and YouTube transcripts."""
from __future__ import annotations

import html
import io
import json
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen as _urlopen

from .db import init_db
from .ingest import ingest_text_document, source_doc_id


GITHUB_API_ROOT = "https://api.github.com"
GITHUB_MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
GITHUB_MAX_EXPANDED_BYTES = 100 * 1024 * 1024
GITHUB_MAX_ARCHIVE_MEMBERS = 10_000
GITHUB_MAX_CONTENT_CHARS = 2_000_000
GITHUB_MAX_FILE_BYTES = 256 * 1024
GITHUB_MAX_FILES = 500
YOUTUBE_MAX_CONTENT_CHARS = 2_000_000
_HTTP_TIMEOUT_SECONDS = 30

_GITHUB_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_YOUTUBE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_TEXT_SUFFIXES = {
    ".bash",
    ".c",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".fish",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".kts",
    ".less",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sass",
    ".scala",
    ".scss",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
_TEXT_FILENAMES = {
    "dockerfile",
    "gemfile",
    "makefile",
    "procfile",
    "requirements.txt",
}


class SourceIngestError(RuntimeError):
    """A source could not be fetched or normalized into an ingest document."""


@dataclass(frozen=True, slots=True)
class SourceDocument:
    doc_id: str
    title: str
    content: str
    url: str


def _parse_github_repository_url(repo_url: str) -> tuple[str, str]:
    parsed = urlsplit(repo_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "GitHub repository URL must be canonical HTTPS: "
            "https://github.com/{owner}/{repo}"
        )

    path = parsed.path.rstrip("/")
    parts = path.split("/")
    if len(parts) != 3 or not parts[1] or not parts[2]:
        raise ValueError(
            "GitHub repository URL must be canonical HTTPS: "
            "https://github.com/{owner}/{repo}"
        )
    owner, repo = parts[1], parts[2]
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    if not owner or not repo or not _GITHUB_NAME.fullmatch(owner) or not _GITHUB_NAME.fullmatch(repo):
        raise ValueError(
            "GitHub repository URL must be canonical HTTPS: "
            "https://github.com/{owner}/{repo}"
        )
    return owner.lower(), repo.lower()


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "deep-research-agent-ingest/2",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_error(repo: str, exc: HTTPError) -> SourceIngestError:
    if exc.code == 404:
        detail = "not found or inaccessible"
    elif exc.code == 409:
        detail = "has no commits"
    elif exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0":
        detail = "rate limit exceeded"
    elif exc.code in {401, 403}:
        detail = "access denied; authenticate with GITHUB_TOKEN if the repository is private"
    else:
        detail = f"GitHub returned HTTP {exc.code}"
    return SourceIngestError(f"GitHub repository {repo} {detail}")


def _github_request_bytes(
    path: str,
    *,
    repo: str,
    max_bytes: int,
) -> bytes:
    request = Request(
        f"{GITHUB_API_ROOT}{path}",
        headers=_github_headers(),
        method="GET",
    )
    try:
        with _urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise SourceIngestError(
                            f"GitHub repository {repo} archive exceeds "
                            f"{max_bytes} bytes"
                        )
                except ValueError:
                    pass
            payload = response.read(max_bytes + 1)
    except HTTPError as exc:
        raise _http_error(repo, exc) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SourceIngestError(
            f"Failed to fetch GitHub repository {repo}: {exc}"
        ) from exc
    if len(payload) > max_bytes:
        raise SourceIngestError(
            f"GitHub repository {repo} archive exceeds {max_bytes} bytes"
        )
    return payload


def _fetch_github_commit(owner: str, repo: str) -> str:
    identity = f"{owner}/{repo}"
    payload = _github_request_bytes(
        f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/commits/HEAD",
        repo=identity,
        max_bytes=256 * 1024,
    )
    try:
        data = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIngestError(
            f"GitHub repository {identity} returned invalid commit metadata"
        ) from exc
    sha = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(sha, str) or not _COMMIT_SHA.fullmatch(sha):
        raise SourceIngestError(
            f"GitHub repository {identity} returned no valid commit hash"
        )
    return sha.lower()


def _fetch_github_archive(owner: str, repo: str, commit_sha: str) -> bytes:
    identity = f"{owner}/{repo}"
    return _github_request_bytes(
        f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/tarball/"
        f"{quote(commit_sha, safe='')}",
        repo=identity,
        max_bytes=GITHUB_MAX_ARCHIVE_BYTES,
    )


def _relative_archive_path(member_name: str) -> PurePosixPath | None:
    parts = PurePosixPath(member_name).parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return None
    return PurePosixPath(*parts[1:])


def _is_supported_text_path(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() in _TEXT_SUFFIXES
        or name in _TEXT_FILENAMES
        or (name.startswith("readme") and (not path.suffix or path.suffix.lower() in _TEXT_SUFFIXES))
    )


def _decode_text(payload: bytes) -> str | None:
    if not payload or b"\x00" in payload:
        return None
    text = payload.decode("utf-8", errors="ignore")
    if not text.strip():
        return None
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
    if printable / len(text) < 0.85:
        return None
    return text.strip()


def _bounded_content(
    prefix: str,
    sections: Iterable[str],
    *,
    limit: int,
    separator: str = "\n\n",
) -> str:
    marker = f"{separator}[truncated: source content limit reached]"
    content = prefix
    truncated = False
    for section in sections:
        addition = (separator if content else "") + section
        if len(content) + len(addition) <= limit:
            content += addition
            continue
        room = max(0, limit - len(content) - len(marker))
        content += addition[:room]
        truncated = True
        break
    if truncated:
        if len(marker) > limit:
            return marker[:limit]
        content = content[: limit - len(marker)] + marker
    return content


def _normalize_github_archive(
    archive_bytes: bytes,
    *,
    owner: str,
    repo: str,
    commit_sha: str,
) -> str:
    identity = f"{owner}/{repo}"
    if len(archive_bytes) > GITHUB_MAX_ARCHIVE_BYTES:
        raise SourceIngestError(
            f"GitHub repository {identity} archive exceeds "
            f"{GITHUB_MAX_ARCHIVE_BYTES} bytes"
        )

    candidates: list[tuple[PurePosixPath, tarfile.TarInfo]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
            expanded_bytes = 0
            for member_number, member in enumerate(archive, start=1):
                if member_number > GITHUB_MAX_ARCHIVE_MEMBERS:
                    raise SourceIngestError(
                        f"GitHub repository {identity} archive contains more than "
                        f"{GITHUB_MAX_ARCHIVE_MEMBERS} entries"
                    )
                if member.isfile():
                    expanded_bytes += member.size
                    if expanded_bytes > GITHUB_MAX_EXPANDED_BYTES:
                        raise SourceIngestError(
                            f"GitHub repository {identity} expanded archive exceeds "
                            f"{GITHUB_MAX_EXPANDED_BYTES} bytes"
                        )
                path = _relative_archive_path(member.name)
                if (
                    not member.isfile()
                    or path is None
                    or member.size <= 0
                    or member.size > GITHUB_MAX_FILE_BYTES
                    or not _is_supported_text_path(path)
                ):
                    continue
                candidates.append((path, member))

            candidates.sort(
                key=lambda item: (
                    0
                    if len(item[0].parts) == 1
                    and item[0].name.lower().startswith("readme")
                    else 1
                    if item[0].name.lower().startswith("readme")
                    else 2,
                    str(item[0]).casefold(),
                    str(item[0]),
                )
            )
            sections: list[str] = []
            seen_paths: set[str] = set()
            files_limited = False
            for path, member in candidates:
                display_path = str(path)
                if display_path in seen_paths:
                    continue
                seen_paths.add(display_path)
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                text = _decode_text(extracted.read(GITHUB_MAX_FILE_BYTES + 1))
                if text is None:
                    continue
                if len(sections) == GITHUB_MAX_FILES:
                    files_limited = True
                    break
                sections.append(f"## File: {display_path}\n\n{text}")
    except (tarfile.TarError, OSError) as exc:
        raise SourceIngestError(
            f"GitHub repository {identity} returned an invalid archive"
        ) from exc

    if not sections:
        raise SourceIngestError(
            f"GitHub repository {identity} contains no supported text files"
        )
    if files_limited:
        sections.append(
            f"[truncated: repository file limit {GITHUB_MAX_FILES} reached]"
        )
    prefix = f"# GitHub repository: {identity}\n\nCommit: {commit_sha}"
    return _bounded_content(
        prefix,
        sections,
        limit=GITHUB_MAX_CONTENT_CHARS,
    )


def load_github_repository(repo_url: str) -> SourceDocument:
    """Fetch and normalize one immutable GitHub commit snapshot."""
    owner, repo = _parse_github_repository_url(repo_url)
    commit_sha = _fetch_github_commit(owner, repo)
    archive_bytes = _fetch_github_archive(owner, repo, commit_sha)
    content = _normalize_github_archive(
        archive_bytes,
        owner=owner,
        repo=repo,
        commit_sha=commit_sha,
    )
    identity = f"{owner}/{repo}@{commit_sha}"
    return SourceDocument(
        doc_id=source_doc_id("github", identity),
        title=f"GitHub {owner}/{repo} @ {commit_sha[:12]}",
        content=content,
        url=f"https://github.com/{owner}/{repo}/tree/{commit_sha}",
    )


def _fetch_youtube_transcript(video_id: str) -> Any:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise SourceIngestError(
            "YouTube ingest requires youtube-transcript-api>=1.2.4"
        ) from exc
    try:
        return YouTubeTranscriptApi().fetch(video_id)
    except Exception as exc:
        raise SourceIngestError(
            f"YouTube transcript unavailable for {video_id}: {exc}"
        ) from exc


def _snippet_text(snippet: Any) -> str | None:
    value = snippet.get("text") if isinstance(snippet, dict) else getattr(snippet, "text", None)
    if not isinstance(value, str):
        return None
    normalized = " ".join(html.unescape(value).split())
    return normalized or None


def load_youtube_transcript(video_id: str) -> SourceDocument:
    """Fetch and normalize the default transcript for one YouTube video ID."""
    if not _YOUTUBE_VIDEO_ID.fullmatch(video_id):
        raise ValueError("YouTube video ID must contain exactly 11 URL-safe characters")
    transcript = _fetch_youtube_transcript(video_id)
    snippets = [text for item in transcript if (text := _snippet_text(item))]
    if not snippets:
        raise SourceIngestError(f"YouTube video {video_id} returned an empty transcript")
    content = _bounded_content(
        "",
        snippets,
        limit=YOUTUBE_MAX_CONTENT_CHARS,
        separator="\n",
    )
    return SourceDocument(
        doc_id=source_doc_id("youtube", video_id),
        title=f"YouTube transcript: {video_id}",
        content=content,
        url=f"https://www.youtube.com/watch?v={video_id}",
    )


def _ingest_source_document(document: SourceDocument) -> int:
    init_db()
    return ingest_text_document(
        doc_id=document.doc_id,
        title=document.title,
        content=document.content,
        url=document.url,
    )


def ingest_github_repository(repo_url: str) -> int:
    """Fetch and atomically ingest one GitHub repository snapshot."""
    return _ingest_source_document(load_github_repository(repo_url))


def ingest_youtube_transcript(video_id: str) -> int:
    """Fetch and atomically ingest one YouTube transcript."""
    return _ingest_source_document(load_youtube_transcript(video_id))
