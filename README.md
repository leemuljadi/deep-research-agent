# Deep-Research Agent

A production-flavoured **multi-agent deep-research pipeline** that answers complex,
multi-step questions by planning, spawning parallel research sub-agents, grounding
answers in a retrieval store, and returning a structured, cited report.

Runs entirely on a **free local stack**, with optional cloud model providers.

---

## What it does

You ask something like:

> *"Compare carbon-accounting obligations for a mid-sized manufacturer across the EU, US and Australia."*

The system:

1. **Plans** the research into a set of sub-questions (lead agent).
2. **Spawns parallel sub-agents** (LangGraph) — each one retrieves + reads sources.
3. **Reflects on research gaps** and runs only novel follow-up questions within a hard cap.
4. **Ingests** retrieved content into a **pgvector** store (chunk → embed → index).
5. **Retrieves** with **hybrid search** (vector + full-text + Reciprocal Rank Fusion).
6. **Synthesizes and reviews** a grounded draft, retaining the best-scoring report.
7. Returns a **structured (Pydantic)**, sourced report.

---

## Capability → module map

| Capability | Where it's implemented |
|---|---|
| Agents planning multi-step tasks over long horizons | `src/graph.py` — plan → parallel researchers → bounded re-plan → synthesis → bounded review |
| Multi-agent / sub-agent orchestration | `src/graph.py` — LangGraph `Send` fan-out spawns one researcher sub-agent per sub-question |
| Stateful LLM workflows and removable reflection edges | `src/graph.py` |
| RAG + retrieval (pgvector, hybrid, re-rank) | `src/db.py`, `src/search.py` (vector + FTS fused by weighted RRF), `src/ingest.py` |
| Structured outputs (function calling / Pydantic) | `src/schemas.py`, tool-call schemas |
| Evaluation harness (accuracy, faithfulness, cost, latency) | `evals/eval_harness.py` — LLM-as-judge accuracy and faithfulness, real token/cost accounting, `--compare` for A/B runs |
| Model routing across providers (LiteLLM) | `src/llm.py` — Router with provider fallback, usage + cost captured per call |
| Observability (OpenTelemetry → Langfuse, tracing) | `src/tracing.py` — OTel spans with token/cost attributes, OTLP export to Langfuse when keys are set |

---

## Architecture (free local stack)

```
src/
├── config.py       # env + settings
├── llm.py          # LiteLLM router (chat + embeddings, multi-provider)
├── tracing.py      # Langfuse / OpenTelemetry hooks
├── schemas.py      # Pydantic output + tool schemas
├── db.py           # pgvector schema + connection
├── ingest.py       # chunk, embed, index
├── search.py       # hybrid search (vector + FTS + RRF)
├── graph.py        # LangGraph deep-research orchestration
└── agents/
    ├── planner.py     # lead agent: plan sub-questions
    ├── researcher.py  # sub-agent: retrieve + read
    └── synthesizer.py # lead agent: write grounded report

evals/
├── eval_harness.py    # batch eval → metrics table
├── judge.py           # LLM-as-judge (faithfulness/accuracy)
└── golden_set.py      # labelled eval questions

scripts/
├── run_research.py    # CLI: answer a question
├── run_eval.py        # CLI: run the eval harness
└── ingest_corpus.py   # CLI: index a document directory
```

**No Azure required.** Postgres+pgvector in Docker, LiteLLM routing to Ollama
(local/free) and optionally to OpenAI/Anthropic. Optionally push vectors into the
**Azure AI Search Free tier** for an Azure touchpoint.

---

## Quick start (free)

```bash
# 1. Start Postgres + pgvector
docker compose up -d db

# 2. Install deps (use a venv)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
#   - set LITELLM_MODEL_CHAT / LITELLM_MODEL_EMBED to your Ollama or cloud models
#   - or leave defaults (Ollama)

# 4. Index the sample corpus
python -m scripts.ingest_corpus data/sample

# 5. Run a research question
python -m scripts.run_research "Compare carbon accounting obligations across EU, US and Australia"

# 6. Run the evaluation harness
python -m scripts.run_eval
```

### With Ollama (100% free, no API keys)

```bash
ollama pull llama3.2        # chat model
ollama pull nomic-embed-text  # embeddings
```

Set in `.env`:
```
LITELLM_MODEL_CHAT=ollama/llama3.2
LITELLM_MODEL_EMBED=ollama/nomic-embed-text
```

### With OpenAI / Anthropic (optional, shows multi-provider routing)

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
LITELLM_MODEL_CHAT=openai/gpt-4o-mini
LITELLM_MODEL_EMBED=openai/text-embedding-3-small
```

LiteLLM routes across whichever providers you configure — set a fallback model to
enable failover:
```
LITELLM_MODEL_CHAT_FALLBACK=anthropic/claude-3-5-haiku-latest
```


---

## Deploy to a VPS

Same stack as local, but on one server: **Caddy → FastAPI (`server.py`) →
Postgres/pgvector (Docker) → LiteLLM → Ollama (host) or OpenAI/Anthropic**.

Files: `Dockerfile`, `docker-compose.yml` (adds `api` + `caddy` services),
`Caddyfile`, `server.py`.

### 1. Prerequisites on the VPS

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
curl -fsSL https://ollama.com/install.sh | sh   # only for the free Ollama path
ollama pull llama3.2
ollama pull nomic-embed-text
```

Ollama must listen on the network interface (not just localhost) so the API
container can reach it:

```bash
# systemd override
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:11434"\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

### 2. Configure and deploy

```bash
git clone <this-repo> && cd deep-research-agent
cp .env.example .env   # defaults already point at Ollama — nothing to change

docker compose up -d --build          # db + api + caddy
docker compose exec api python -m scripts.ingest_corpus data/sample   # index corpus
```

`.env` values are read by compose (`${VAR:-default}`) and passed through to the
API container. `OLLAMA_API_BASE` defaults to `http://host.docker.internal:11434`,
which reaches the host-run Ollama via the `host-gateway` mapping.

### 3. Use

**Web UI** — open `http://<vps-ip>/` (or `http://localhost:8000` when running
`uvicorn server:app` locally): type a question, get the report with findings,
cited sources and run metrics.

**HTTP API** — same report as JSON:

```bash
curl http://<vps-ip>/ping
curl -X POST http://<vps-ip>/research -H 'content-type: application/json' \
  -d '{"question":"Compare carbon accounting obligations across EU, US and Australia"}'
```

### HTTPS

Edit `Caddyfile`: replace `:80` with your domain (DNS A record → VPS first) and
`docker compose restart caddy`. Caddy provisions and renews Let's Encrypt
certificates automatically.

### Ops notes

- **VPS size**: the pipeline is I/O-bound (LLM + DB). 2 vCPU / 4 GiB RAM is fine
  for Ollama with `llama3.2` (3B); go bigger for 7B+ models.
- **Data**: `dra_pgdata` volume holds documents + vectors; survives container
  restarts. Back up with `docker compose exec db pg_dump dra > backup.sql`.
- **Embedding consistency**: index the corpus with the *same* embed model the
  API uses (`LITELLM_MODEL_EMBED` + `EMBEDDING_DIM`), or vector search breaks.
- **No public DB port**: only Caddy (80/443) and SSH should be reachable;
  Postgres is exposed on 5432 to the host network for local dev — on the VPS,
  consider removing the `ports:` block from the `db` service (compose services
  talk over the internal network; `expose:` is enough).

---


## Bounded reflection (optional)

The graph owns two small, removable refinement loops: planner re-planning after
a research round and draft review before completion. Both stop at their
non-negative cap, on repeated work, or when the score fails to improve by more
than the comparison epsilon. Earlier candidates win ties, and the highest
scoring plan/report is retained.

```bash
# .env defaults: at most one additional research round and one draft revision
REFLECTION_ENABLED=true
PLANNER_REFLECTION_MAX_ITERATIONS=1
SYNTHESIS_REVIEW_MAX_ITERATIONS=1

# deletion-equivalent / AD-11 loop-off baseline
REFLECTION_ENABLED=false
```

Caps count **additional** rounds or revisions and accept zero. Reflection calls
are skipped entirely when disabled (or when the corresponding cap is zero).
Values are frozen into each run's state, so a worker restart or HITL resume
cannot change a run halfway through. The synthesis gate fires only after review
selects the best report.

---


## Per-run cost cap (optional)

Every run can be bounded by a hard spend cap (AD-15). Crossing it terminates
the run loudly — status `cost_cap_exceeded`, no degraded report, no soft cap:

```bash
# .env — unset by default (uncapped)
RUN_COST_CAP_USD=2.00
```

| Aspect | Behavior |
|---|---|
| **Scope** | Per run. Each run accumulates its own total; one run's spend never affects another. |
| **Default** | `RUN_COST_CAP_USD` unset = **uncapped** — the free Ollama path never trips a cap (its cost stays $0). |
| **Per-run override** | Set `cost_cap_usd` on a `research_jobs` row (nullable column) — it wins over the env default. |
| **Enforcement** | One home: the run-scoped accumulator in `src/llm.py`. Every LLM/chat/embedding capture (and, from story 6, every tool cost) folds into the shared run-total and is checked immediately. On the worker's stepwise path, overshoot is bounded by one in-flight cost-bearing call; the CLI's compiled-graph fan-out runs calls concurrently, so its bound is the cost of the calls in flight. |
| **Exactly reached** | Trips (`>=` semantics) — no free overage. |
| **Trip behavior** | The run ends with status `cost_cap_exceeded`; the cost summary (cap, accumulated total) is recorded in the run's `error` field. Partial state is not persisted — the trip is terminal. The worker moves on to the next job. |
| **Bad values** | An unparseable `RUN_COST_CAP_USD` (e.g. `cheap`) logs a startup warning and runs uncapped — it never crashes the worker. |

**Suggested range:** for cloud-model runs on this repo's corpus, start at
**$0.50–$5.00 per run** — a single web-search-class call can cost ~$0.01–0.10,
and open_deep_research logged $45.98–$187.09 per 100-task benchmark run
(langchain-ai/open_deep_research cost logs, 2025-08). Treat the number as a
circuit breaker, not a budget: set it comfortably above a normal run's measured
cost so only runaway loops trip it. There is no verified benchmark for this
corpus yet — instrument real per-run cost (eval harness) before tightening.

---

## Evaluation harness

`evals/eval_harness.py` runs a labelled **golden set** through the full pipeline and
reports:

| Metric | Meaning |
|---|---|
| **Accuracy** | LLM-as-judge: fraction of ground-truth concepts substantively covered (lexical fallback if judge unavailable) |
| **Faithfulness** | LLM-as-judge: fraction of findings SUPPORTED by the cited source snippets |
| **Cost** | Real token cost per run from LiteLLM usage (estimated only for unpriced local models) |
| **Tokens** | Total prompt + completion tokens across every agent call |
| **Latency** | End-to-end wall-clock time per query |

Outputs a markdown table to stdout and a JSON report to `evals/reports/`.

```bash
python -m scripts.run_eval --name loop-off --loop-mode off
python -m scripts.run_eval --name loop-on --loop-mode on
python -m scripts.run_eval --compare loop-off loop-on
python -m evals.gate --shadow loop-off loop-on
```

`--loop-mode` is explicit in paired measurements: both reports use the same
golden samples and the existing shadow gate keeps its story 7 exit-code
contract. `off` makes one planner, one research round, and one synthesis call
with no reflection LLM calls; `on` uses the frozen configured caps.

---

## Project structure notes

- **Well-typed tool schemas** — every agent tool is a Pydantic model (`schemas.py`)
  so tool-call arguments are validated and testable.
- **Evaluation-first** — the harness is the entry point, not an afterthought, so you
  can iterate on prompts/chunking/models with measurable signals.
- **Clean, maintainable** — small modules, type hints, and no framework lock-in
  beyond LangGraph + LiteLLM.
