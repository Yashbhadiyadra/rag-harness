# Architecture

## Overview

A reliability-first RAG pipeline over the pinned Kubernetes documentation.
The point is not just to answer questions — it is to **measure** answer
quality across each independent failure mode, catch regressions before
they ship, and expose the same measurement story to the visitor at the
public demo.

Six modules, each with a single responsibility, composed behind a
FastAPI service and a static demo UI. Every non-trivial decision is
recorded as an ADR; every metric is written to append-only history and
rendered on a public metrics page.

## Modules

| Module | Responsibility | Key ADRs |
|---|---|---|
| `ingest` | Load pinned K8s docs, clean, chunk on headings, embed via OpenAI, index in ChromaDB with provenance | ADR-0001, ADR-0002, ADR-0003, ADR-0005 |
| `retrieval` | Query → candidate chunks via one of five composable strategies (dense, hybrid, hybrid-rerank, hyde, full) | ADR-0006 |
| `generation` | Retrieved context + query → grounded answer via `gpt-4o-mini`. Optional corrective critic-and-retry loop | ADR-0007 |
| `evaluation` | Score answers on five metrics; enforce reliability gate; append-only run history; ablation runner | ADR-0004 |
| `observability` | Per-call token/cost tracking (ContextVar collectors); OpenTelemetry spans exported to Arize Phoenix; Prometheus counters | ADR-0008, ADR-0009 |
| `api` | FastAPI service (`/query`, `/health`, `/ready`, `/metrics`) with middleware chain (kill switch, daily cap, rate limit), demo UI at `/`, and CLI subcommands | ADR-0010 |

## Data flow (ingest + query)

```
K8s docs repo (pinned git commit — ADR-0002)
        │
        ▼
    [ingest]  load markdown → heading-aware chunk (ADR-0003)
              → OpenAI embeddings (cached in SQLite — ADR-0005)
              → upsert into ChromaDB with provenance (ADR-0001)
        │
        ▼
    ChromaDB persistent store  ┐
                               │
    BM25 in-memory index  ─────┤
                               │
   query ─────────────────────▶│
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                              ▼
    [retrieval — ADR-0006]                     [generation — ADR-0007]
    Optional HyDE query expansion              Grounded answer via gpt-4o-mini
    Dense / hybrid (dense + BM25 RRF)          Optional corrective loop:
    Optional cross-encoder rerank                critic scores each chunk;
                                                 reformulate + retry, or
                                                 return honest refusal
        │                                              │
        └──────────────────────┬──────────────────────┘
                               ▼
                     [evaluation — ADR-0004]
                     context_recall + context_precision
                     + faithfulness + correctness
                     + answer_relevancy
                     ↓
                     append to evals/history/runs.jsonl
```

## Request path (runtime)

Every `POST /query` traverses a defined middleware chain before the
handler runs. Order is intentional (ADR-0010): the cheapest gate first,
the LLM boundary last.

```
POST /query
    ↓
KillSwitchMiddleware        DEMO_ENABLED=false → 503 demo_disabled
    ↓                       (health/ready/metrics stay reachable)
DailyCapMiddleware          ≥ 200/day globally → 429 demo_daily_limit_reached
    ↓                       (in-memory counter, single writer under max-instances=1)
SlowAPIMiddleware           > 10/hour or > 3/minute per IP → 429
    ↓
FastAPI validation          pydantic guardrails (question length, top_k range)
    ↓
Input guardrail             regex hygiene for prompt-injection patterns
    ↓
collect_spans() + collect_usage()   observability collectors opened
    ↓
Retrieval                   selected strategy — dense / hybrid / hyde / …
    ↓
Generation                  gpt-4o-mini (or corrective loop)
    ↓
Response                    { answer, sources, trace, cost_usd, latency_ms }
```

The response includes the per-stage trace and this-query cost/latency
because the demo UI renders them next to the answer (ADR-0010).

## Resilience layer

Every I/O-bound function is `async`. The LLM boundary is wrapped in an
`AsyncOpenAI` client with `max_retries=2` and `timeout=20s`, plus an
outer honest-refusal fallback. Failure modes route as follows:

| Failure | Outcome | HTTP | Body |
|---|---|---|---|
| Rate limit hit (per-IP) | slowapi rejection | 429 | slowapi default |
| Daily cap hit (global) | middleware rejection | 429 | `demo_daily_limit_reached` |
| Kill switch on | middleware rejection | 503 | `demo_disabled` |
| Prompt-injection pattern | guardrail rejection | 422 | `guardrail_rejection` |
| ChromaDB unreachable | readiness check fails; existing requests retry | 503 (`/ready` only) | `not_ready` |
| OpenAI retries exhausted | honest refusal | **200** | `NO_INFO_MESSAGE`, empty sources |
| Anything else | last-resort 500 | 500 | internal_error |

The OpenAI-exhausted path is deliberately 200: the API contract stays
useful, the visitor sees "not enough information" rather than a scary
5xx, and the `rag_query_errors_total` counter still fires so operators
see the degradation.

## Observability layer

Three parallel channels feed different consumers:

1. **In-request span collector** (`observability/tracing.collect_spans()`)
   — a `ContextVar`-based accumulator that captures every `traced_span`
   closure and is returned on the `QueryResponse.trace` field. Runs
   independently of Phoenix so the demo UI works whether or not a
   tracing backend is reachable.
2. **OpenTelemetry export to Arize Phoenix** (ADR-0009) — the same
   `traced_span` calls emit OTel spans to the Phoenix backend when
   `TRACING_ENABLED=true`. Waterfall UI, prompt/completion side-by-side,
   trace search. Optional in the deployed image.
3. **Prometheus counters + histograms** (`api/metrics.py`) —
   `rag_query_total`, `rag_query_errors_total{error_type}`,
   `rag_query_tokens_total{direction,model}`, `rag_query_cost_usd_total`,
   `rag_query_latency_seconds` histogram. Scraped from `/metrics`.

Per-call cost accounting uses the same `ContextVar` pattern
(`observability/usage.collect_usage()`) and returns per-request
`cost_usd` alongside the trace (ADR-0008).

## Evaluation layer

- **Golden set** — hand-verified cases in `evals/golden/`, organised by
  topic. Version-controlled and reviewed like code.
- **Five metrics** (ADR-0004):
  - Retrieval: `context_recall` (deterministic set intersection),
    `context_precision` (LLM judge).
  - Grounding: `faithfulness` (LLM judge).
  - Generation: `correctness` (LLM judge), `answer_relevancy` (LLM judge).
- **Ablation runner** sweeps every `(strategy, corrective)` pair and
  emits a comparative table (`evals/experiments/`); a highlighted
  "relevant but incorrect" column tracks confident hallucination.
- **Append-only history** — every eval and ablation run appends a
  `HistoryEntry` to `evals/history/runs.jsonl`. That file feeds the
  static metrics page (`scripts/render_metrics_page.py`).
- **Reliability gate** — the eval suite fails when any metric drops
  below its configured threshold. Run on every PR (5-case subset), full
  suite nightly, and on every release tag before deploy.

## Cost guardrails

Documented in [ADR-0010](adr/ADR-0010-cloud-run-and-persistence.md);
enforced across three independent gates:

- **Per-IP rate limit** — `slowapi` with composite
  `10/hour;3/minute`. Configurable via `API_RATE_LIMIT`.
- **Global daily cap** — 200 requests/day across all IPs, enforced by
  the in-memory `DailyBudget` under `max-instances=1`. Resets at 00:00
  UTC. Configurable via `DEMO_DAILY_REQUEST_CAP`.
- **Monthly budget ceiling** — Cloud Billing budget with alerts at
  50%, 90%, and 100% of `$10/month`.
- **Emergency kill switch** — `DEMO_ENABLED=false` env var → instant
  503 on `/query`; probes stay reachable.

## Deploy pipeline

```
git tag v*.*.*
    ↓
.github/workflows/release.yml
    ↓
make check        →  full 30-case eval gate  →  WIF auth to GCP
    ↓                                              ↓
                                             docker build with baked chroma_db
                                             + OCI provenance labels
                                              ↓
                                             push to Artifact Registry
                                              ↓
                                             sed-substitute manifest placeholders
                                              ↓
                                             gcloud run services replace
                                              ↓
                                             verify latestReady == latestCreated
                                             (Cloud Run rollback if not)
                                              ↓
                                             smoke-test /health with retries
```

Complete runbook in [`deploy/README.md`](../deploy/README.md).

## Forward compatibility (Projects 2 and 3)

This is Project 1 of a three-part reliability portfolio. Later projects
depend on choices made here:

- **Project 2 (stale-embedding detection)** consumes ingest history —
  every chunk records `source_file`, `git_commit`, and `doc_version` as
  provenance. Never discard that; it is Project 2's input signal.
- **Project 3 (agent trajectory evaluator)** reuses the `evaluation`
  layer — metric interfaces and the `HistoryEntry` schema are kept
  generic, not hard-wired to single-turn RAG only.
- **MCP server (stretch)** — the FastAPI surface is deliberately narrow
  and stable; exposing it as MCP tools (`query_docs`, `run_ablation`) is
  a thin wrapper away.

## Design decisions

All non-trivial decisions are recorded as ADRs in
[`docs/adr/`](adr/):

| ADR | Decision |
|---|---|
| [ADR-0001](adr/ADR-0001-chromadb.md) | ChromaDB as vector store |
| [ADR-0002](adr/ADR-0002-pinned-corpus.md) | Corpus pinned to immutable git commit |
| [ADR-0003](adr/ADR-0003-heading-based-chunking.md) | Heading-based chunking |
| [ADR-0004](adr/ADR-0004-llm-as-judge-evaluation.md) | LLM-as-judge for semantic metrics |
| [ADR-0005](adr/ADR-0005-embedding-cache.md) | SQLite embedding cache |
| [ADR-0006](adr/ADR-0006-hybrid-retrieval-and-reranking.md) | Hybrid + rerank + HyDE composition |
| [ADR-0007](adr/ADR-0007-corrective-rag.md) | Corrective critic-and-retry loop |
| [ADR-0008](adr/ADR-0008-observability-usage-tracking.md) | ContextVar token/cost tracking |
| [ADR-0009](adr/ADR-0009-tracing-backend.md) | Arize Phoenix for per-stage tracing |
| [ADR-0010](adr/ADR-0010-cloud-run-and-persistence.md) | Cloud Run, baked Chroma, cost guardrails |
