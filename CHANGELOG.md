# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
