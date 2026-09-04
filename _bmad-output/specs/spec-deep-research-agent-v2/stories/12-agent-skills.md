---
title: 'Agent Skills distribution channel (CAP-10)'
type: 'feature'
status: 'draft'
baseline_commit: '1162e81482f5ac598e18df94b303df6e9b0ab15b'
route: 'in-session'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-deep-research-agent-v2/SPEC.md'
  - '{project-root}/_bmad-output/planning-artifacts/architecture/architecture-deep-research-agent-2026-09-02/ARCHITECTURE-SPINE.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Claude Code, Cursor, and Codex users have no packaged instructions for driving the research pipeline. Without a governed skill, a harness may import `src.graph`, call an agent directly, or invent a control surface that bypasses the async job boundary and forks from the MCP contract.

**Approach:** Ship an instruction-only `skills/` distribution channel for Claude Code, Cursor, and Codex. Each `SKILL.md` exposes exactly the AD-17 job-boundary verbs `submit`, `poll`, `approve`, `redirect`, and `cancel`. Synchronous submission uses only `python -m scripts.run_research`; asynchronous operation uses the existing `server.py` HTTP routes. A repository test parses those route decorators and proves that every packaged skill documents the same endpoints, verb set, and no-bypass rule. No tools, helper scripts, services, dependencies, or runtime code are added.

## Boundaries & Constraints

**Always:** package skills beneath repository-root `skills/`; include valid Agent Skills `name` and `description` frontmatter; provide install and invocation instructions for Claude Code, Cursor, and Codex; name all five job-boundary verbs (`submit`, `poll`, `approve`, `redirect`, `cancel`); document synchronous submit through `python -m scripts.run_research "<question>"`; document asynchronous submit/poll/control through the routes actually declared in `server.py`; preserve `run_id` between asynchronous operations; tell harnesses to poll until a terminal state or `waiting_for_input`; explain the redirect request shape; keep the skill and MCP verb sets equal by an explicit test contract; remain free-local compatible.

**Never:** no `src/`, `server.py`, worker, queue, schema, database, MCP, dependency, or deployment changes; no executable scripts inside skills; no direct imports or invocation of `src.graph`, `src.agents`, graph nodes, database transition functions, or any pipeline stage from a harness; no synchronous graph execution except through the documented CLI entrypoint; no endpoints or verbs beyond the five AD-17 job/run transitions; no claim that the synchronous CLI returns a `run_id` or supports poll/gate controls; no dependency on story 11 implementation code.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Synchronous submit | A quoted non-empty question; repository environment configured | Harness runs `python -m scripts.run_research "<question>"` from the repository root and consumes the printed report | Non-zero exit is surfaced; harness never imports or calls `src.graph` directly |
| Asynchronous submit | `POST /research` with JSON `{"question":"..."}` | Response supplies `run_id`; harness stores it unchanged for later verbs | HTTP or JSON failure is surfaced; no fallback to direct graph execution |
| Poll active run | `GET /runs/{run_id}` returns `queued` or `running` | Harness waits and polls the same run without resubmitting | 404 is surfaced as an unknown run; no fabricated state |
| Poll gate | Poll returns `waiting_for_input` | Harness reports the gate and chooses only approve, redirect, or cancel from user intent | Harness does not approve or redirect speculatively when user intent is unavailable |
| Approve | `POST /runs/{run_id}/approve` with an empty body | Waiting run returns to `queued`; harness resumes polling | 409 is surfaced as an invalid/raced transition |
| Redirect | `POST /runs/{run_id}/redirect` with `{"decision":"redirect","payload":{"question":"..."}}` | Waiting run returns to `queued` with the replacement question; harness resumes polling | Empty question produces 422; 409 is surfaced as an invalid/raced transition |
| Cancel | `POST /runs/{run_id}/cancel` with an empty body | Queued or waiting run becomes `cancelled` | Running or otherwise non-cancellable run returns 409 and is surfaced |
| Terminal poll | Poll returns `completed`, `failed`, `cancelled`, or `cost_cap_exceeded` | Harness stops polling; returns `report` on completion or the terminal status/error otherwise | No retry loop after a terminal state |
| Channel drift | A skill verb or endpoint differs from the server/MCP contract | `tests/test_skills.py` fails before merge | Update the packaging docs to the stable API contract; do not add runtime surface |
| Boundary bypass request | Prompt asks the harness to call a graph node, agent, or DB transition directly | Harness refuses the bypass and uses the CLI or HTTP job boundary | If neither boundary is available, stop and report the missing prerequisite |

## Open Questions

- [ASSUMPTION] Ship one harness-labelled skill per consumer (`skills/claude-code/`, `skills/cursor/`, `skills/codex/`) rather than one shared folder. This makes installation explicit while the parity test prevents content drift.
- [ASSUMPTION] Examples default `DRA_BASE_URL` to `http://localhost:8000`; deployments override it with their API origin. Authentication headers remain deployment-specific because the current stable `server.py` API contract defines none.
- [ASSUMPTION] Skills are instruction-only and copied or symlinked into each harness's supported project/user skill directory. No installer script is warranted for three files.
- [ASSUMPTION] Story 11's MCP implementation is not imported or inspected. The five-verb MCP contract is pinned from CAP-8/AD-17 and compared to the skill verbs explicitly until story 11 is available on the integration branch.

</frozen-after-approval>

## Code Map

- `skills/claude-code/deep-research-agent/SKILL.md` (new) -- Claude Code-facing job-boundary workflow; same five verbs, CLI command, HTTP routes, terminal-state handling, and no-bypass rule.
- `skills/cursor/deep-research-agent/SKILL.md` (new) -- Cursor-facing job-boundary workflow with the identical contract.
- `skills/codex/deep-research-agent/SKILL.md` (new) -- Codex-facing job-boundary workflow with the identical contract.
- `skills/README.md` (new) -- install locations and invocation examples for all three harnesses; explains repository-root source packaging versus harness discovery paths.
- `tests/test_skills.py` (new) -- parse `server.py` with `ast`; map the five job-boundary verbs to route method/path pairs; parse every distributed `SKILL.md`; assert required files, portable frontmatter, exact verbs/endpoints, documented CLI entrypoint, and no-bypass language. Pin `MCP_JOB_BOUNDARY_VERBS` explicitly to CAP-8/AD-17 and assert each skill's set equals it.

## Tasks & Acceptance

**Execution:**
- [x] Add Claude Code, Cursor, and Codex `SKILL.md` packages with identical job-boundary semantics.
- [x] Document synchronous CLI submission and all five asynchronous HTTP operations using only existing paths.
- [x] Add an explicit prohibition on direct graph, agent, database-transition, and pipeline-stage invocation.
- [x] Add `skills/README.md` with project and user installation paths plus harness invocation syntax.
- [x] Add `tests/test_skills.py` to derive server routes from AST and enforce route/MCP/skill parity.

**Acceptance Criteria:**
- Given any shipped `SKILL.md`, when its declared job-boundary verbs are parsed, then the set is exactly `submit`, `poll`, `approve`, `redirect`, and `cancel`.
- Given `server.py`, when its FastAPI route decorators are parsed, then the method/path mapped to each skill verb exists exactly as documented: `POST /research`, `GET /runs/{run_id}`, and `POST /runs/{run_id}/approve|redirect|cancel`.
- Given the CAP-8/AD-17 MCP contract, when compared with each skill, then both channels expose the same five verbs.
- Given any shipped skill, when scanned, then it instructs the harness to never bypass the job boundary and forbids direct graph/agent invocation outside `python -m scripts.run_research`.
- Given a supported harness, when the README installation step is followed, then the skill lands in that harness's documented discovery directory and can be invoked by its packaged name.
- Given the offline suite, when run, then all tests pass without runtime-source changes.

## Verification

**Commands:**
- `python -m unittest tests.test_skills` -- route, endpoint, verb, frontmatter, CLI, and no-bypass contract checks pass.
- `python -m scripts.run_research` -- prints the repository's usage line without starting a run; confirms the documented module path. The CLI has no dedicated help flag on the baseline.
- `python -m unittest discover -s tests` -- full offline suite passes.
- `git diff -- src server.py scripts` -- empty; confirms packaging-only scope.

## Suggested Review Order

1. `tests/test_skills.py` -- the channel-consistency and no-bypass guarantees.
2. One `skills/*/SKILL.md` -- command shapes and workflow semantics.
3. The remaining skills -- harness-specific invocation wording only; contract tables remain identical.
4. `skills/README.md` -- installation paths and usage.
