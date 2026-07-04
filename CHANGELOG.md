# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
