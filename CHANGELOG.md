# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (Phase 11 — pre-v1.0 audit + hardening)
- **Secret + dep-vulnerability scanning in CI.** New parallel
  ``security`` job in ``ci.yml`` runs on every push and PR:
  ``gitleaks`` with ``fetch-depth: 0`` scans the full commit history
  for exposed secrets; ``pip-audit --strict --disable-pip`` scans
  installed runtime deps for known vulnerabilities. Top-level
  ``permissions: contents: read`` limits both jobs to the minimum.

## [0.9.0] — 2026-07-06

### Added (Phase 10 — deployment, demo UI, metrics page)
- **ADR-0010: Cloud Run + scale-to-zero + baked Chroma.** Records the
  hosting choice (Cloud Run over Fly.io / Render / self-managed VM),
  persistence strategy (bake the 83 MB ``chroma_db/`` into the image
  over rebuild-on-boot or mounted storage), the ``max-instances=1 +
  in-memory counter`` architecture that keeps the daily-cap counter a
  single writer, and the exact guardrail numbers.
- **Global daily request cap.** New ``DailyBudget`` class
  (``src/rag_harness/api/budget.py``) with thread-safe counter and
  UTC-day rollover. ``DailyCapMiddleware`` consumes one slot per
  ``POST /query`` and returns HTTP 429 with a
  ``demo_daily_limit_reached`` body when exhausted. Default cap 200
  requests/day; configurable via ``DEMO_DAILY_REQUEST_CAP``.
- **Emergency kill switch.** ``KillSwitchMiddleware`` returns HTTP 503
  ``demo_disabled`` on ``/query`` when ``DEMO_ENABLED=false``.
  ``/health``, ``/ready``, ``/metrics`` stay reachable in every state
  so operators can still probe the service.
- **Per-stage trace, cost, and latency on the ``/query`` response.**
  New ``collect_spans()`` ContextVar collector in
  ``observability/tracing.py`` runs independently of Phoenix and
  captures every ``traced_span`` closure. ``QueryResponse`` gains
  ``trace: list[TraceSpan]``, ``cost_usd: float``, and
  ``latency_ms: float`` — the data the demo UI renders under the
  answer.
- **Minimal demo UI at ``/``.** Single static bundle under
  ``src/rag_harness/api/static/`` — HTML, CSS, and vanilla JS with no
  build step. Question form, answer + sources with heading-path
  breadcrumbs, a per-stage trace waterfall (bars sized proportional
  to max span duration), and a cost/latency footer. Dark mode via
  ``prefers-color-scheme``. Distinct friendly error banners for
  ``demo_daily_limit_reached``, ``demo_disabled``, per-IP 429,
  ``guardrail_rejection``, and ``not_ready``. ``noindex,nofollow`` on
  the page keeps search bots off the daily cap.
- **HEAD ``/`` support.** ``@app.api_route("/", methods=["GET","HEAD"])``
  so uptime monitors (which default to HEAD) do not false-alert.
- **Static metrics-page generator.**
  ``scripts/render_metrics_page.py`` reads
  ``evals/history/runs.jsonl`` and renders a single self-contained
  ``docs/metrics/index.html`` (no CDN, no JS, no CSS framework):
  latest-run headline with production-config callout; ablation table
  with inline SVG sparklines per metric column; quality-vs-cost
  scatter (one labelled dot per ``(strategy, corrective)`` combo);
  and an "Impact of corrective RAG" panel citing ADR-0007 that
  colours positive correctness deltas green and negative ones red.
- **Cloud Run deploy manifest + runbook.**
  ``deploy/cloud-run.yaml`` (Knative Service with ADR-0010's config:
  ``min=0, max=1, concurrency=40, memory=1Gi, cpu=1, timeout=60s``,
  ``OPENAI_API_KEY`` from Secret Manager, ``/health`` startup + liveness
  probes). ``deploy/README.md`` walks through the one-time GCP setup:
  project + APIs, Artifact Registry, Secret Manager, runtime + deploy
  service accounts, Workload Identity Federation scoped to the repo,
  ``$10/month`` Cloud Billing budget with 50/90/100% alerts, first
  manual deploy + probe checklist, and teardown.
- **Release-tag CD pipeline.** ``.github/workflows/release.yml`` on
  ``v*.*.*`` tag push: ``make check`` → full 30-case eval gate →
  WIF-auth to GCP → docker build with OCI provenance labels → push to
  Artifact Registry → ``sed``-substitute placeholders in the
  manifest → ``gcloud run services replace`` → verify latest revision
  reached ``ready`` (Cloud Run's startup probe handles rollback; the
  workflow fails loud if it did) → smoke-test ``/health`` with retries.
  Concurrency-serialized; deploys land through a "production" GitHub
  environment for optional manual approval.
- **Metrics-page regen workflow.**
  ``.github/workflows/metrics-page.yml`` triggers on ``Nightly Eval``
  completion (success) or manual dispatch; regenerates
  ``docs/metrics/index.html`` and commits with ``[skip ci]`` if
  changed.
- **Nightly eval commits history rows.**
  ``.github/workflows/eval.yml`` gains ``permissions: contents: write``
  and a final step that commits new rows of
  ``evals/history/runs.jsonl`` so the metrics-page workflow has fresh
  data on the next trigger.
- **Demo documentation.** ``docs/DEMO.md`` is the demo's public
  reference: live URL (placeholder until first tagged release), a
  table mapping each UI element to the ADR that motivated it,
  guardrail rationale with worst-case cost math, cap-tripped state
  glossary, local-reproduction steps, deploy pointer, and
  cross-references to every relevant ADR.

### Changed
- **Public per-IP rate limit tightened.** Default was ``60/minute``
  (sized for local dev). New default is the composite
  ``10/hour;3/minute`` (see ADR-0010). Autouse conftest fixture resets
  the limiter between tests so the tighter limit does not leak state
  across test files. ``API_RATE_LIMIT`` override still supported for
  local dev.
- **Dockerfile: multi-stage bake + Cloud Run polish.** Builder stage
  installs ``.[eval]`` only (rerank and observability extras omitted
  to keep the runtime image slim). ``chroma_db/`` is copied into
  ``/app/chroma_db`` in the runtime stage as a build input
  (``make docker-build`` runs ``make ingest`` first when missing).
  ``VOLUME`` declarations dropped. Non-root ``app`` user. CMD switched
  to shell form with ``exec`` and ``${PORT:-8000}`` so Cloud Run's
  PORT injection interpolates and SIGTERM reaches uvicorn as PID 1.
- **``.dockerignore`` no longer excludes ``chroma_db/``.** It is now
  a build input rather than a runtime-mounted volume.
- **New Makefile targets.** ``docker-build`` and ``docker-run`` (for
  local end-to-end verification); ``metrics-page`` (regenerate the
  metrics page from local eval history).
- **README public-demo section.** New section between "Why" and
  "Retrieval strategies" summarising the guardrails and linking to
  ``docs/DEMO.md``. Rate-limiting bullet in "Production hardening"
  updated to the new default. Design-decisions list extended with
  ADR-0010.
- **Package version bumped to 0.9.0** in ``pyproject.toml`` to match
  this changelog entry. The first git tag ever (``v0.1.0``) will be
  cut only after this audit and the first live deploy are both done;
  ``v1.0.0`` follows after the Stage D statistical rigor work
  (bootstrap CIs, golden-set expansion, judge calibration).

## [0.8.0] — 2026-07-05

### Added (Phase 9 — production hardening)
- **Async request path end-to-end.** Every I/O-bound function now uses
  ``AsyncOpenAI`` and ``await``; the FastAPI /query handler is
  ``async def``. Retrieval, generation, corrective, HyDE, critic, and
  the eval + ablation runners all have first-class ``_async`` variants
  with sync facades for the CLI boundary. Ingest embedder gains
  bounded-concurrency batching via ``asyncio.Semaphore`` (default 4).
- **Retries + timeouts at the LLM boundary.** New
  ``observability/openai_client.build_async_client()`` returns an
  ``AsyncOpenAI`` configured with ``openai_max_retries=2`` and
  ``openai_timeout_seconds=20.0``. Combined worst case ~40 s per call
  before we degrade to the honest refusal.
- **Graceful refusal on total LLM failure.** ``OpenAIError`` in the
  /query handler returns 200 with ``NO_INFO_MESSAGE`` and an empty
  sources list; the ``rag_query_errors_total`` counter and a
  ``logger.warning`` still fire so ops can see the degradation.
- **Rate limiting.** ``slowapi`` (core dep) with in-memory per-IP
  ``60/minute`` default limiter. Configurable via ``API_RATE_LIMIT``.
- **Request size caps.** ``QueryRequest.question`` gains
  ``min_length=1, max_length=2000`` (``API_MAX_QUESTION_LENGTH``);
  ``top_k`` gains ``ge=1, le=50``.
- **Minimal prompt-injection screening.**
  ``src/rag_harness/api/guardrails.py`` — regex-based screening for the
  most common patterns (``ignore previous instructions``, ``you are
  now …``, ``<system>`` tags, etc.). Explicit non-goal: this is a
  hygiene layer, not a full guardrails engine.
- **Typed exception hierarchy.**
  ``src/rag_harness/api/errors.py`` — ``RagHarnessError`` root with
  ``GuardrailRejection``, ``RetrievalError``, ``GenerationError``, and
  ``NotReadyError`` subclasses; each maps to the correct HTTP status
  and a structured JSON body via a single FastAPI exception handler.
- **``/ready`` endpoint** distinct from ``/health``. ``/health`` stays
  trivial (liveness only); ``/ready`` checks ChromaDB heartbeat,
  ``OPENAI_API_KEY`` presence, and (strict) cross-encoder importability
  when the strategy needs it.
- **Load-check script + baseline results.** ``scripts/load_check.py``
  boots the FastAPI app in-process with mocks, fires N concurrent
  requests, measures p50/p95/p99 latency and throughput at multiple
  concurrency levels, writes a markdown table to
  ``docs/load-check/<ts>.md``. First run (10/25/50/100 concurrent,
  200 ms injected LLM latency) records 100% success and near-linear
  throughput up to 100 concurrent — see
  ``docs/load-check/20260705T050836Z.md``.

### Changed
- ``retriever.retrieve`` becomes a sync facade over the new
  ``retrieve_async`` abstract method on every Retriever implementation.
- LLM-judge metrics (``faithfulness``, ``correctness``,
  ``answer_relevancy``, ``context_precision``) gain ``_async``
  variants; sync facades kept.
- ``QueryRequest`` field ``question`` renamed on the handler signature
  to ``body`` (``request: fastapi.Request`` is required by slowapi).
- ``NO_INFO_MESSAGE`` is now the graceful-degradation path for both
  the corrective refusal and any post-retry LLM failure.

## [0.7.0] — 2026-07-04

### Added (Phase 8 — evaluation completeness + ablation study)
- Corrective RAG wired into `run_eval` behind a keyword-only
  `use_corrective` flag (also `--corrective` on the CLI's `eval`
  subcommand). `EvalResult` gains `corrective_category`,
  `corrective_attempts`, `corrective_reformulated_query`.
- SQLite LLM judge response cache (`observability/llm_cache.py`) —
  wraps `faithfulness`, `correctness`, `answer_relevancy`,
  `context_precision`. **Not** wired to `generate()` or the corrective
  critic (deliberate footgun avoidance). Cache hits skip the API and
  record no token usage. Off by default.
- Ablation runner (`evaluation/ablation.py`) — sweeps every strategy in
  `VALID_STRATEGIES` × [baseline, corrective] and emits a single
  comparative markdown table + full CSV. New CLI: `rag-harness ablation
  [--output-dir evals/experiments] [--no-cache]`.
- **Relevant-but-incorrect** as a first-class output. Cases where
  `answer_relevancy > 0.7` AND `correctness < 0.5` — confident-sounding
  hallucination — get their own column in the ablation markdown, their
  own row in the terminal summary, and a per-case boolean flag in the
  CSV.
- Append-only eval history at `evals/history/runs.jsonl`. One line per
  `run_eval` invocation and per ablation configuration. Grows with git
  history so quality drift and cost blowups are attributable to
  specific commits.
- Per-PR reliability gate — `.github/workflows/eval-pr.yml` runs the
  gate against a 5-case subset on every PR to main. `~$0.02 per PR`.
  Guarded so PRs from forks can't spend the repo's `OPENAI_API_KEY`.
  Full suite still runs nightly.
- `EVAL_PR_SUBSET_IDS`, `RBI_RELEVANCY_MIN`, `RBI_CORRECTNESS_MAX`,
  `LLM_CACHE_ENABLED`, `LLM_CACHE_PATH` config keys.
- CLI `--subset pr|id1,id2,...` on the `eval` subcommand.

### Changed
- `run_eval` grows keyword-only `strategy_label`, `record_history`, and
  `case_filter` parameters. Default behaviour unchanged.

## [0.6.0] — 2026-07-03

### Added
- Observability foundation (Phase 7): model rates pricing table
  (`observability/pricing.py`), immutable `TokenUsage` record, and a
  `collect_usage()` ContextVar collector for opting into per-block token
  and cost tracking without changing existing function signatures.
- `record_usage()` calls at every OpenAI boundary — chat completions in
  generator/critic/corrective-reformulation/HyDE, query embeddings in the
  dense retriever, batched chunk embeddings in the ingest embedder.
- Two new LLM-as-judge metrics: `answer_relevancy` (is the answer
  on-topic?) and `context_precision` (fraction of retrieved chunks that
  materially support the reference answer). Complete the five-metric suite.
- Extended `EvalResult` with `latency_ms`, `input_tokens`, `output_tokens`,
  `estimated_cost_usd`. Extended `EvalSummary` with `mean_context_precision`,
  `mean_answer_relevancy`, `latency_p50_ms`, `latency_p95_ms`,
  `total_cost_usd`, `total_input_tokens`, `total_output_tokens`.
- `python -m rag_harness eval` now prints quality + operational metrics in
  one unified table (Phase 7 exit criterion).
- `GET /metrics` Prometheus endpoint on the API server with counters for
  query volume, errors, tokens (by direction/model), and cost; histogram
  for latency by strategy. Default process/GC collectors are unregistered
  so the endpoint is RAG-focused.
- Per-stage tracing (Arize Phoenix backend, see ADR-0009). New
  `observability/tracing.py` with a `configure_tracing()` bootstrapper
  and a `traced_span()` context manager that is a cheap no-op when
  disabled. Stages instrumented: `evaluate_case`, `retrieve`, `generate`,
  `score` in the eval runner; `query`, `retrieve`, `generate`,
  `corrective_generate` on the API server. OpenAI SDK calls
  auto-instrumented via `openinference-instrumentation-openai` so they
  nest inside stage spans.
- `docker-compose.yml` gains a `phoenix` service (image
  `arizephoenix/phoenix:latest`, UI on `:6006`).
- `MODEL_RATES_OVERRIDES`, `TRACING_ENABLED`, `TRACING_ENDPOINT`,
  `TRACING_SERVICE_NAME` config keys.
- Optional `[observability]` extra: `arize-phoenix-otel`,
  `openinference-instrumentation-openai`. Base install stays lightweight.
- ADR-0008: rationale for ContextVar over signature changes.
- ADR-0009: Phoenix vs Langfuse comparison; **Phoenix chosen** on
  operational fit, Cloud Run alignment, and reversibility.

## [0.5.0] — 2026-07-02

### Added
- **Corrective RAG** — critic-and-retry loop from Yan et al. 2024. The
  pipeline now judges its own retrieval quality:
  - `RelevanceCritic` scores every retrieved chunk in a single `gpt-4o-mini`
    call using structured JSON output; scores are calibrated on a rubric
    (1.0 = directly answers → 0.0 = irrelevant).
  - Three-way routing: Correct (max score ≥ 0.7) → filter and generate;
    Ambiguous (0.3 ≤ max < 0.7) → filter and generate anyway; Incorrect
    (max < 0.3) → reformulate the query and retry once.
  - Final fallback returns the byte-identical "not enough information"
    refusal used by the generator, keeping evaluation signals consistent.
- `CorrectiveResult` dataclass carries answer plus telemetry (category,
  attempts, per-chunk scores, reformulated query) for downstream
  observability and ablation.
- Config fields: `CORRECTIVE_RAG_ENABLED`, `CRITIC_CORRECT_THRESHOLD`,
  `CRITIC_INCORRECT_THRESHOLD`, `CORRECTIVE_MAX_RETRIES`.
- `--corrective` flag on the `query` CLI subcommand.
- API server: `POST /query` accepts an optional `corrective` field; falls
  back to `CORRECTIVE_RAG_ENABLED` when unset.
- ADR-0007 documenting the design, threshold choices, cost tradeoffs,
  and rejected alternatives.

### Changed
- Critic always scores against the ORIGINAL query, not the reformulated
  one — reformulation is a search tool, relevance judgement stays anchored.
- Reformulation failure falls back to the original query rather than
  aborting the retry loop.

## [0.4.0] — 2026-07-02

### Added
- **Hybrid retrieval** — `HybridRetriever` combines dense semantic search with
  BM25 sparse search via Reciprocal Rank Fusion (score = Σ 1/(k + rank + 1),
  k=60 per Cormack et al. 2009). Rank-based fusion avoids score normalisation.
- **BM25 sparse index** — `BM25Store` builds an in-memory `BM25Okapi` index
  from all ChromaDB documents at startup; tokeniser deliberately simple to
  preserve K8s identifiers like `PodDisruptionBudget` and `kubectl`.
- **Cross-encoder reranker** — `RerankingRetriever` wraps any base retriever
  with a second-stage rerank using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  Gated behind the optional `[rerank]` extra (`sentence-transformers`,
  ~500MB with PyTorch).
- **HyDE query transformation** — `HyDERetriever` asks `gpt-4o-mini` to draft
  a hypothetical answer, then uses that as the retrieval query; bridges the
  query-answer vocabulary gap. Falls back to raw query on LLM failure.
- **Strategy factory** — `build_retriever(strategy)` composes retrievers by
  name. Strategies: `dense`, `hybrid`, `hybrid-rerank`, `hyde`, `full`.
- `--strategy` flag on both `rag-harness query` and `rag-harness eval`
  subcommands; also configurable via `RETRIEVAL_STRATEGY` in `.env`.
- API server picks its retriever via `RETRIEVAL_STRATEGY`.
- ADR-0006 documenting the hybrid retrieval, reranking, and HyDE design.

### Changed
- Core deps: added `rank_bm25>=0.2.2` (pure Python, 50KB)
- API server no longer hard-codes `DenseRetriever`; uses `build_retriever`
  with the configured strategy

## [0.3.0] — 2026-07-01

### Added
- 30 hand-written golden evaluation cases across five K8s topic areas:
  workloads, networking, storage, scheduling, and cluster operations
- End-to-end integration test suite (`tests/integration/`) using real ChromaDB
  and mocked OpenAI; excluded from per-PR CI, run with `pytest -m integration`
- `pytest.mark.integration` marker registered in `pyproject.toml`

## [0.2.0] — 2026-07-01

### Added
- SQLite embedding cache (`EmbeddingCache`) — re-ingesting an unchanged corpus
  makes zero OpenAI API calls after the first run; path configurable via
  `EMBEDDING_CACHE_PATH`
- Eval results export — `python -m rag_harness eval --output results.json` (or
  `.csv`) saves per-case scores for regression tracking across nightly runs
- Centralised logging configuration (`logging_setup.py`) with timestamp and
  logger-name format; log level configurable via `LOG_LEVEL` in `.env`
- FastAPI lifespan hook applies logging config at server startup
- Multi-stage Dockerfile and `docker-compose.yml` for local deployment; ChromaDB
  and embedding cache mounted as named volumes
- `make ingest`, `make serve`, `make eval` targets
- ADR-0005: SQLite embedding cache design decision

### Changed
- Module-level and class-level docstrings added across all packages

## [0.1.0] — 2026-06-30

### Added
- Phase 1 discipline: pinned K8s corpus to `snapshot-initial-v1.32`
  (SHA `bbb60b97`), two new ADRs, full CHANGELOG backfill, docstrings
  on all public functions

## [0.1.0] — 2026-06-30

### Added
- FastAPI server (`POST /query`, `GET /health`) and CLI (`ingest`, `query`, `eval`
  subcommands); `python -m rag_harness` entry point
- Three-metric evaluation layer: deterministic context recall, LLM-as-judge
  faithfulness and correctness; reliability gate fails CI when any metric drops
  below threshold
- Grounded answer generator using `gpt-4o-mini` at `temperature=0`; context-only
  system prompt enforces faithfulness contract
- Dense retriever backed by ChromaDB cosine similarity; abstract `Retriever`
  interface keeps generation and evaluation decoupled from the vector store
- Ingest pipeline: heading-aware markdown chunker preserving heading hierarchy as
  provenance, OpenAI embedder with 512-item batching, idempotent ChromaDB indexer
  storing full provenance metadata per chunk
- Shared data models: `Chunk` (with provenance), `GoldenCase`, `EvalResult`,
  `EvalSummary`; Pydantic settings with `.env` support
- 23 unit tests across all modules; `make check` gate (ruff + mypy + pytest)
- First hand-written golden eval case (RBAC); `evals/golden/` directory
- Architecture doc and ADRs: ChromaDB choice, pinned corpus, heading-based
  chunking, LLM-as-judge evaluation
