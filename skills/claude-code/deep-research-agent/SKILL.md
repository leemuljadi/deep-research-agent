---
name: deep-research-agent
description: Submit and control deep-research-agent runs from Claude Code through the supported CLI or HTTP job boundary. Use for research requests, run status, gate approval, redirect, or cancellation.
---

# Deep Research Agent

Drive the repository's research pipeline without crossing its serving boundary.

## Boundary contract

**NEVER bypass the job boundary.** Use only the documented synchronous CLI entrypoint or the HTTP operations below. Do not import or invoke `src.graph`, `src.agents`, graph nodes, pipeline stages, or database transition functions from Claude Code or from harness-authored scripts. The only permitted synchronous graph invocation is behind `python -m scripts.run_research`.

If the CLI or API is unavailable, report that prerequisite instead of falling back to an internal function. Do not add a new endpoint or control verb.

## Job-boundary verbs

- `submit`
- `poll`
- `approve`
- `redirect`
- `cancel`

This set is exact. It is the same job-boundary verb set exposed by the MCP channel.

## Supported HTTP contract

| Verb | HTTP method | Route |
|------|-------------|-------|
| `submit` | `POST` | `/research` |
| `poll` | `GET` | `/runs/{run_id}` |
| `approve` | `POST` | `/runs/{run_id}/approve` |
| `redirect` | `POST` | `/runs/{run_id}/redirect` |
| `cancel` | `POST` | `/runs/{run_id}/cancel` |

## Choose one submission mode

### Synchronous CLI submit

From the repository root, run:

```bash
python -m scripts.run_research "Compare the evidence for the proposed policy"
```

The quoted argument is the research question. This foreground command prints the report and does not return a reusable `run_id`; therefore `poll`, `approve`, `redirect`, and `cancel` do not apply to this mode. Surface any non-zero exit to the user.

### Asynchronous HTTP submit

Use this mode when the run must survive the initiating command or may require gate control. Set the API origin once:

```bash
DRA_BASE_URL="${DRA_BASE_URL:-http://localhost:8000}"
```

Submit a non-empty question:

```bash
curl --fail-with-body --silent --show-error \
  -X POST "$DRA_BASE_URL/research" \
  -H 'content-type: application/json' \
  -d '{"question":"Compare the evidence for the proposed policy"}'
```

Read `run_id` from the JSON response and retain it exactly:

```bash
RUN_ID="<run_id from submit response>"
```

## Poll

```bash
curl --fail-with-body --silent --show-error \
  "$DRA_BASE_URL/runs/$RUN_ID"
```

Interpret `status` as follows:

- `queued` or `running`: wait before polling the same `RUN_ID` again; never resubmit merely because work is pending.
- `waiting_for_input`: report the gate and obtain user intent before `approve`, `redirect`, or `cancel`.
- `completed`: stop polling and return `report`.
- `failed`, `cancelled`, or `cost_cap_exceeded`: stop polling and surface the terminal status and `error` when present.

A 404 means the run is unknown. Surface it; do not fabricate status or start an internal run.

## Approve

Only after the run reports `waiting_for_input` and the user intends approval:

```bash
curl --fail-with-body --silent --show-error \
  -X POST "$DRA_BASE_URL/runs/$RUN_ID/approve"
```

On success the run is `queued`; resume polling. Surface 409 as an invalid or raced transition.

## Redirect

Only after the run reports `waiting_for_input` and the user supplies a replacement question:

```bash
curl --fail-with-body --silent --show-error \
  -X POST "$DRA_BASE_URL/runs/$RUN_ID/redirect" \
  -H 'content-type: application/json' \
  -d '{"decision":"redirect","payload":{"question":"Investigate the replacement question"}}'
```

On success the run is `queued`; resume polling. A blank replacement returns 422. Surface 409 as an invalid or raced transition.

## Cancel

Only when the user requests cancellation:

```bash
curl --fail-with-body --silent --show-error \
  -X POST "$DRA_BASE_URL/runs/$RUN_ID/cancel"
```

Cancellation is valid for queued or waiting runs. A running or otherwise non-cancellable run returns 409; surface it without forcing an internal state change.
