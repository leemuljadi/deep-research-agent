---
title: 'MCP mount with job-boundary verbs (CAP-8)'
type: 'feature'
status: 'draft'
baseline_commit: '1162e81482f5ac598e18df94b303df6e9b0ab15b'
route: 'in-session'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/architecture-deep-research-agent-2026-09-02/ARCHITECTURE-SPINE.md'
---

<frozen-after-approval reason="human-owned intent and defaults — do not modify unless human renegotiates">

## Intent

**Problem:** the API already exposes asynchronous research jobs and human gate transitions, but Claude, Cursor, and other MCP clients cannot drive that lifecycle. Exposing graph execution directly as an MCP tool would bypass the Postgres job boundary, block requests through multi-round runs, and violate AD-14/AD-17.

**Approach:** Add a purpose-built FastMCP 4 server with exactly five tools — `submit`, `poll`, `approve`, `redirect`, and `cancel` — mounted at `/mcp` inside the existing FastAPI process. Each tool is a thin wrapper over the same `src/db.py` job-boundary functions used by `server.py`; no tool imports or executes the graph or an agent. Use stateless HTTP, chain the API and MCP lifespans, and keep one container and one port. When `MCP_JWT_SECRET` is set, protect the mount with FastMCP `JWTVerifier` using HS256; when unset, leave auth disabled for development and emit a startup warning.

## Boundaries & Constraints

**Always:** FastMCP 4 is built in `src/mcp_server.py`; `server.py` mounts `mcp.http_app(path="/", stateless_http=True)` at `/mcp`; API startup/database initialization and MCP session-manager lifespans are chained in entry order and unwind in reverse order; the tool list is exactly `{submit, poll, approve, redirect, cancel}`; tools call only `enqueue_job`, `get_job`, `record_decision`, `resume_run`, and `cancel_run`; `poll` maps a row to `RunStatusResponse`; cross-boundary request/response shapes live in `src/schemas.py` (AD-6); `MCP_JWT_SECRET` flows through `src/config.py`; configured auth is `JWTVerifier(public_key=secret, algorithm="HS256")`; malformed, wrong-signature, and expired JWTs are rejected by FastMCP before a tool call; missing auth configuration is allowed only as the documented development posture and logs a startup warning; `mcp>=2` and the FastMCP 4 release line are declared dependencies.

**Never:** no MCP tool imports `src.graph`, `src.agents`, or `src.tools`; no synchronous graph or agent execution; no new SQL or database transition; no sixth convenience/debug/admin tool; no second service, container, or port; no Redis/Celery/RabbitMQ; no OAuth/OAuthProxy or external-MCP consumption; no changes to `src/graph.py` or `src/tools.py`; no auth secret in source, logs, or response payloads; no network in tests.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Submit | non-blank question | `enqueue_job(question)` once; returns queued run response immediately | DB error propagates as MCP tool error; graph is never called |
| Submit invalid | empty/whitespace question | Pydantic input validation rejects before enqueue | MCP validation/tool error; zero DB calls |
| Poll found | valid run id and queued/running/waiting/terminal row | `get_job(run_id)` once; returns `RunStatusResponse`, validating any completed report | corrupt stored report fails loudly; no raw row leaks |
| Poll unknown/malformed | absent row or invalid UUID | not-found tool error | malformed ids inherit `get_job`'s no-DB guard |
| Approve waiting run | row exists; decision and resume writes win | `get_job` → `record_decision(..., "approve", payload)` → `resume_run`; queued transition response | missing run = not found; stale/double/raced transition = conflict tool error |
| Redirect waiting run | row exists; non-blank replacement question | preserves decision payload for audit; trims question passed to `resume_run`; queued transition response includes normalized question | missing/blank question = validation tool error before writes; stale/raced transition = conflict tool error |
| Cancel queued/waiting run | row exists and conditional cancel wins | `get_job` → `cancel_run`; cancelled transition response | unknown = not found; running/terminal/raced state = conflict tool error |
| Tool inventory | in-memory client lists tools | names are exactly the five job-boundary verbs | any extra tool fails the contract test |
| Auth configured | valid HS256 bearer signed with `MCP_JWT_SECRET` | request reaches MCP | malformed, wrong-signature, or expired JWT returns unauthorized and does not call a tool |
| Auth unset | development environment omits `MCP_JWT_SECRET` | MCP remains usable without a token | API startup emits an explicit auth-disabled warning |
| Route coexistence | `/mcp` mounted with existing `/runs/{id}` routes | both route families remain reachable; mount consumes only `/mcp` | path collision/order regression fails route tests |
| Lifespan | API starts/stops | API lifespan enters before MCP lifespan and exits after it; DB initialization remains once | missing MCP lifespan fails mounted transport smoke test |

## Open Questions

- **JWT issuer/audience constraints:** [ASSUMPTION] CAP-8 specifies only one shared-secret env key, so v2 validates HS256 signature plus standard temporal claims without introducing issuer/audience configuration. Add claim restrictions only when an issuer contract exists.
- **Unset-secret environment detection:** [ASSUMPTION] absence always disables MCP auth and warns; deployment documentation makes setting a strong secret mandatory before proxy exposure rather than guessing a separate production-mode flag.
- **Mutation error vocabulary:** [ASSUMPTION] MCP surfaces not-found, invalid-input, and transition-conflict conditions as `ToolError` messages matching REST semantics; the wire protocol owns the JSON-RPC error envelope.
- **Redirect payload shape:** [ASSUMPTION] expose `redirect(run_id, question)` as the focused MCP interface and construct the existing `GateDecision` payload internally; no arbitrary client payload is needed for the sole redirect transition.

</frozen-after-approval>

## Code Map

- `src/mcp_server.py` (new) -- `build_mcp_server(jwt_secret)` constructs the optional `JWTVerifier`, registers exactly five thin synchronous tools, maps job rows into `RunStatusResponse`, and exposes the module-level FastMCP server plus stateless ASGI app. It imports `src.db` as a module so tool names cannot shadow transition functions.
- `src/schemas.py` -- move the submit shape out of `server.py` and add named run-accepted/run-transition response models shared by REST and MCP; reuse `GateDecision`, `RunStatus`, `RunStatusResponse`, and `ResearchReport`.
- `src/config.py` -- add `mcp_jwt_secret` sourced only from `MCP_JWT_SECRET` (AD-9).
- `server.py` -- use shared schemas, emit the unset-secret warning during API startup, chain the existing lifespan with the MCP app lifespan, and mount only at `/mcp`; existing `/research` and `/runs/*` handlers keep their database paths.
- `requirements.txt` -- add `fastmcp>=4,<5` and `mcp>=2` (AD-12 protocol floor).
- `.env.example`, `docker-compose.yml`, and `README.md` -- pass through and document private-deployment JWT setup, the unsafe development-only auth-disabled posture, and proxy/protocol hardening.
- `tests/test_mcp_server.py` (new) -- FastMCP in-memory client inventory/call tests with mocked DB functions; direct `JWTVerifier` rejection tests for malformed, wrong-signature, and expired tokens; mount/lifespan/static route assertions; source-boundary regression preventing graph/agent execution.
- `tests/test_requirements.py` -- assert the `mcp>=2` floor and FastMCP 4 release line.

## Tasks & Acceptance

**Execution:**
- [ ] Add shared Pydantic submit and transition response shapes in `src/schemas.py`; migrate `server.py` to them.
- [ ] Add `MCP_JWT_SECRET` to `src/config.py`, `.env.example`, the API container environment, and README deployment guidance.
- [ ] Build `src/mcp_server.py` with exactly five wrappers over existing `src/db.py` functions.
- [ ] Mount the stateless MCP ASGI app at `/mcp` and combine lifespans without changing the API/worker deployment shape.
- [ ] Add FastMCP 4 and `mcp>=2` dependency floors, including requirement assertions.
- [ ] Add mocked, in-memory FastMCP tests for every verb, auth failures, inventory, mount isolation, and the no-graph boundary.

**Acceptance Criteria:**
- Given an in-memory FastMCP client, when it lists tools, then the result is exactly `submit`, `poll`, `approve`, `redirect`, and `cancel`, with no graph-stage or shell-executing tool.
- Given each valid verb call, when the database boundary is mocked, then the corresponding existing `src/db.py` function sequence is called exactly once and the result validates against a schema in `src/schemas.py`; no SQL is added.
- Given any MCP tool module/import path, when inspected and exercised, then `src.graph`, `src.agents`, and `src.tools` are absent and no graph/agent runs synchronously.
- Given `MCP_JWT_SECRET`, when the mount receives a malformed, wrong-signature, or expired token, then FastMCP rejects it before any tool executes; a valid HS256 token is accepted.
- Given `MCP_JWT_SECRET` unset, when the API lifespan starts, then auth is disabled and one explicit startup warning states that `/mcp` is unauthenticated.
- Given the API process, when routes are inspected and exercised, then `/mcp` coexists with `/runs/{id}` and `/research`; the API lifespan initializes the database and the MCP lifespan is entered and exited in the documented order; deployment remains one container and one port.
- Deployment hardening is documented: expose no shell-executing tools; run Uvicorn with `--proxy-headers` behind the trusted proxy; disable response buffering for `/mcp`; test both legacy and 2026-07-28 protocol eras during migrations.
- Given dependency checks, `requirements.txt` contains `fastmcp>=4,<5` and `mcp>=2`, and `tests/test_requirements.py` rejects a lower MCP floor.
- Given the offline suite, all tests pass with FastMCP calls using `fastmcp.Client` in memory and no network.

## Implementation Notes

## Spec Change Log

## Review Triage Log

## Design Notes

- Mounting uses `mcp.http_app(path="/", stateless_http=True)` plus `app.mount("/mcp", mcp_app)` so the public endpoint is `/mcp` rather than `/mcp/mcp`; this also isolates the mount from `/runs/*`.
- `combine_lifespans(api_lifespan, mcp_app.lifespan)` preserves database initialization before MCP request serving and reverses cleanup order.
- Authentication is server-construction configuration, not per-tool code. This prevents a verb from accidentally omitting authorization.

## Verification

**Commands:**
- `uv run python -m unittest tests.test_mcp_server tests.test_requirements` -- focused contract and dependency checks.
- `uv run python -m unittest discover -s tests` -- full offline suite.
- `uv run pip check` -- dependency graph is consistent.
- `uv run python -c "import server; import src.mcp_server; assert any(getattr(r, 'path', None) == '/mcp' for r in server.app.routes)"` -- import and mount check.

## Suggested Review Order

**The protocol boundary**

- `src/mcp_server.py` -- inventory, database-only calls, error mapping, JWTVerifier construction.
- `server.py` -- stateless mount, route ordering, warning, and combined lifespan.

**Contracts and regression cover**

- `src/schemas.py` -- shared boundary shapes.
- `tests/test_mcp_server.py` and `tests/test_requirements.py` -- five verbs, auth failures, no graph execution, and dependency floors.
