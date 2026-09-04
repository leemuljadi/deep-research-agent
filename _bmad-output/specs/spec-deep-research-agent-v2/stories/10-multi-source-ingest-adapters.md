---
title: 'Multi-source ingest adapters — GitHub and YouTube (CAP-7)'
type: 'feature'
created: '2026-09-04'
status: 'draft'
baseline_commit: '1162e81482f5ac598e18df94b303df6e9b0ab15b'
route: 'in-session'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-deep-research-agent-v2/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-deep-research-agent-v2/stories.yaml'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** corpus ingest accepts only local `.txt`, `.md`, and `.py` files. CAP-7 requires operators to ingest GitHub repository snapshots and YouTube transcripts without adding either source to the research agent's runtime tool surface. Story 9 established the atomic replacement path required to prevent stale or orphaned chunks during re-ingest.

**Approach:** add ingest-time source adapters behind `scripts/ingest_corpus`. A GitHub adapter resolves a public or token-authorized repository URL to its current commit, downloads one bounded archive, and normalizes README plus supported text/code files into one deterministic snapshot document. A YouTube adapter fetches one transcript by video ID and normalizes snippets into one text document. Both use the existing chunk/embed contract and persist exclusively through `src.db.replace_document`.

## Boundaries & Constraints

**Always:** source IDs use the existing lowercase SHA-1/16 rule with a source namespace: `sha1("github:{owner}/{repo}@{commit_sha}")[:16]` for a GitHub snapshot and `sha1("youtube:{video_id}")[:16]` for a YouTube transcript; GitHub owner/repository identity is case-normalized and the resolved commit hash is included; repeated ingestion of the same identity calls `replace_document` with the same document ID, atomically replacing every prior chunk; chunk rows are ordered `(chunk_index, text, embedding)` triples created by the existing chunker/embedder contract; GitHub fetches use HTTPS and honor `GITHUB_TOKEN` when present; source/network failures happen before any document replacement; tests fake all network and database boundaries.

**Never:** never call legacy `upsert_document` or `insert_chunks`; never place SQL outside `src/db.py` (AD-3); never add GitHub or YouTube as runtime tools and never modify `src/tools.py` or `src/graph.py`; never clone repositories or extract archives to disk; never ingest binary files, archive links, device entries, or unbounded source data; never issue live HTTP or database calls from tests.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| GitHub happy path | canonical `https://github.com/{owner}/{repo}` URL; commit and archive available | resolve commit SHA; README first, then deterministic supported code/text files; write one snapshot document and return its chunk count | N/A |
| GitHub idempotent re-ingest | same repository resolves to same commit | same SHA-1/16 document ID; one atomic replacement; no prior chunks survive | N/A |
| GitHub new commit | same repository resolves to a different commit | distinct immutable snapshot ID containing the new commit identity | previous snapshot remains addressable; no cross-document deletion |
| Empty repository | GitHub reports no commits or archive contains no supported non-empty text | no document write | raise a source-specific ingest error |
| Private/missing repository | GitHub returns 401/403/404 | no document write; message names repository and status without exposing token | raise a source-specific ingest error; 403 identifies rate-limit/auth possibility |
| Huge repository/archive | compressed response or normalized content exceeds bounds | bounded read; deterministic README-first/file-path selection; normalized content ends with a truncation marker when the text budget is reached | oversized archive is rejected before parsing; eligible content is capped |
| YouTube happy path | valid 11-character video ID with transcript | normalize non-empty snippets to newline-delimited text; write one transcript document and return chunk count | N/A |
| YouTube idempotent re-ingest | same video ID | same SHA-1/16 document ID; one atomic replacement; no prior chunks survive | N/A |
| Transcript unavailable | disabled captions, removed/private video, request block, or provider error | no document write | raise a source-specific ingest error preserving the cause as text |
| Empty transcript | fetch succeeds with no non-whitespace snippets | no document write | raise a source-specific ingest error |
| Invalid CLI/source input | missing value, unknown flag, malformed GitHub URL, malformed video ID | print usage/input error; no DB/network call | CLI exits non-zero or adapter raises `ValueError` before I/O |
| Embedding mismatch/invalid vector | embedder returns wrong count or `replace_document` validation fails | no partial corpus state | propagate validation error; story 9 transaction remains authoritative |

## Open Questions

- [ASSUMPTION] A GitHub commit is one corpus document, not one document per file. This makes the commit-hash identity literal, bounds replacement to one transaction, and matches gitingest-style concatenated output.
- [ASSUMPTION] GitHub accepts only canonical HTTPS repository-root URLs; branch/file URLs and arbitrary Git hosts are rejected rather than ambiguously normalized.
- [ASSUMPTION] GitHub normalized text is capped at 2,000,000 characters, individual files at 256 KiB, at most 500 eligible files, and archive download at 20 MiB. README files are selected first, then lexical repository paths; truncation is explicit.
- [ASSUMPTION] YouTube accepts an 11-character video ID, not a URL, and stores the provider's default transcript language without translation.
- [ASSUMPTION] `youtube-transcript-api>=1.2.4` is the smallest justified dependency because YouTube exposes no stable public transcript endpoint; import remains lazy so local-file/GitHub ingest does not require the optional path at module import time.

</frozen-after-approval>

## Code Map

- `src/ingest.py` -- add the shared namespaced SHA-1/16 helper and a text-document ingest function that chunks, embeds, constructs ordered triples, and calls only `replace_document`; keep local-file IDs backward compatible.
- `src/ingest_sources.py` (new) -- source validation, bounded GitHub HTTP/archive normalization, YouTube transcript normalization, source-specific errors, and top-level GitHub/YouTube ingest entry points.
- `scripts/ingest_corpus.py` -- preserve positional directory mode; add exact `--github <url>` and `--youtube <video-id>` modes using the current `sys.argv` convention.
- `requirements.txt` -- add only `youtube-transcript-api>=1.2.4`; GitHub uses stdlib `urllib`, `json`, and `tarfile`.
- `tests/test_ingest_sources.py` (new) -- network-faked adapter unit tests for IDs, normalization, bounds, errors, idempotent replacement arguments, and no-write failure paths.
- `tests/test_ingest_cli.py` (new) -- mocked CLI dispatch/usage tests with no network or database.
- `tests/test_ingest.py` -- shared text-document contract tests through `replace_document`.
- `tests/test_requirements.py` -- assert the transcript dependency floor and document why it is required.

## Tasks & Acceptance

**Execution:**
- [ ] Write failing tests for namespaced IDs, shared text ingest, GitHub archive normalization/error cases, YouTube transcript normalization/error cases, and CLI dispatch.
- [ ] Add shared source-document ingestion while preserving local-file ID behavior.
- [ ] Implement bounded GitHub commit/archive adapter with optional token authentication.
- [ ] Implement lazy-import YouTube transcript adapter and normalized text output.
- [ ] Wire `--github` and `--youtube` into `scripts/ingest_corpus` without touching runtime tools.
- [ ] Add the transcript dependency floor and requirement policy assertion.
- [ ] Run focused tests, self-review edge cases, then run the full offline verification set.

**Acceptance Criteria:**
- Given a GitHub repository whose resolved commit is unchanged, when it is ingested twice, then both writes use `sha1("github:{owner}/{repo}@{sha}")[:16]` and each call supplies the complete current chunk set to `replace_document`.
- Given a bounded fake GitHub archive, when ingested, then one normalized document contains README before lexically sorted supported code/text files and excludes binary/unsupported entries.
- Given an empty, missing, private, rate-limited, or oversized GitHub repository response, when ingested, then a useful source error is raised and neither `init_db` nor `replace_document` is called.
- Given a valid video ID and fake transcript, when ingested, then the document ID is `sha1("youtube:{video_id}")[:16]`, snippet text is normalized, and ordered schema-conform chunks reach `replace_document`.
- Given an unavailable or empty transcript, when ingested, then a useful source error is raised and no DB write occurs.
- Given CLI directory, `--github`, and `--youtube` inputs, when `main()` runs, then exactly the matching ingest path executes; malformed modes exit non-zero before I/O.
- Given the repository test suite, when run offline, then no test performs live HTTP or database access and all tests pass.

## Verification

**Commands:**
- `uv run python -m unittest tests.test_ingest tests.test_ingest_sources tests.test_ingest_cli tests.test_requirements` -- focused adapter/contract/dependency tests, all mocked.
- `uv run python -m unittest discover -s tests` -- full offline suite passes.
- `uv run pip check` -- installed dependency graph is consistent.
- `uv run python -c "import src.ingest_sources; import scripts.ingest_corpus; print('ingest adapters OK')"` -- both source adapters and CLI import without network/DB activity.

## Suggested Review Order

1. Frozen ID, atomicity, and bounded-input contract.
2. `src/ingest_sources.py` network/error boundaries and deterministic normalization.
3. Shared `src/ingest.py` replacement path and local-file ID compatibility.
4. CLI dispatch and network-faked tests.
