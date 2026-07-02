# rag-harness

[![CI](https://github.com/Yashbhadiyadra/rag-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashbhadiyadra/rag-harness/actions/workflows/ci.yml)

A reliability-first Retrieval-Augmented Generation (RAG) system over the
[Kubernetes documentation](https://github.com/kubernetes/website). The goal is
not just to answer questions — it is to **measure** answer quality independently
across each failure mode and catch regressions before they ship.

## Why

A RAG pipeline has three independent failure modes. Scoring each one
independently lets you pinpoint exactly what broke — not just that
"something went wrong":

| Stage | Failure | Metric | Threshold |
|---|---|---|---|
| Retrieval | Wrong chunks fetched | Context Recall | ≥ 0.80 |
| Generation | Answer not grounded in context | Faithfulness | ≥ 0.85 |
| Generation | Answer is factually wrong | Correctness | ≥ 0.75 |

The evaluation suite runs against a hand-verified golden set of 30 cases
spanning workloads, networking, storage, scheduling, and cluster operations.
A build fails when any metric drops below its threshold.

## Retrieval strategies

Five composable retrieval strategies, selectable via `--strategy` or the
`RETRIEVAL_STRATEGY` environment variable:

| Strategy | Composition | Best for |
|---|---|---|
| `dense` | OpenAI embeddings + ChromaDB cosine similarity | Baseline; semantic questions |
| `hybrid` | Dense + BM25 fused with Reciprocal Rank Fusion (k=60) | Mixed queries with exact-term identifiers |
| `hybrid-rerank` | Hybrid retrieval + cross-encoder reranker | Precision-critical use |
| `hyde` | Hypothetical Document Embeddings on top of dense | Query-answer vocabulary mismatch |
| `full` | HyDE ∘ RerankingRetriever ∘ HybridRetriever | Highest quality; highest cost |

Every strategy is documented in [ADR-0006](docs/adr/ADR-0006-hybrid-retrieval-and-reranking.md)
with the tradeoffs and alternatives considered.

### Corrective RAG (opt-in)

An optional critic-and-retry loop wraps any retrieval strategy. The critic scores
each retrieved chunk for relevance and routes the pipeline:

- **Correct** — filter weak chunks, generate the answer
- **Ambiguous** — filter, then generate on the smaller surviving set
- **Incorrect** — reformulate the query and retry once; refuse rather than
  hallucinate if the second attempt also fails

Enable with `--corrective` on the CLI, `corrective: true` in the API body, or
`CORRECTIVE_RAG_ENABLED=true` in `.env`. Details and cost tradeoffs in
[ADR-0007](docs/adr/ADR-0007-corrective-rag.md).

## Setup

**Requirements:** Python 3.12, an OpenAI API key.

```bash
git clone https://github.com/Yashbhadiyadra/rag-harness.git
cd rag-harness

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"

# Optional: cross-encoder reranker (adds ~500 MB with PyTorch)
# pip install -e ".[rerank]"

pre-commit install
cp .env.example .env
# add your OPENAI_API_KEY to .env
```

## Usage

```bash
# Ingest the pinned Kubernetes docs snapshot (clone → chunk → embed → index).
# Re-runs are free — the embedding cache skips previously-seen chunks.
make ingest

# Ask a question with the default (dense) strategy
python -m rag_harness query "How do I configure RBAC in Kubernetes?"

# Same question with hybrid retrieval + cross-encoder reranking
python -m rag_harness query "How do I configure RBAC in Kubernetes?" \
    --strategy hybrid-rerank

# Run the golden eval suite and enforce the quality gate
make eval

# Save per-case scores for regression tracking
python -m rag_harness eval --output results.json    # or .csv
```

Serve the API:

```bash
make serve
# POST /query  {"question": "...", "top_k": 5}
# GET  /health
```

Deploy with Docker Compose:

```bash
docker compose up -d
# API on http://localhost:8000
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  K8s docs repo  │────▶│      ingest      │────▶│  ChromaDB   │
│  (pinned SHA)   │     │  load → chunk →  │     │  (dense)    │
└─────────────────┘     │  embed → index   │     └─────────────┘
                        └──────────────────┘            │
                                │                       │
                                ▼                       ▼
                        ┌──────────────────┐     ┌─────────────┐
                        │ embedding cache  │     │  BM25Store  │
                        │    (SQLite)      │     │ (in-memory) │
                        └──────────────────┘     └─────────────┘
                                                        │
              query ────────────────────────────────────┼──────┐
                │                                       │      │
                ▼                                       ▼      ▼
        ┌─────────────┐     ┌──────────────┐     ┌────────────────┐
        │    HyDE     │────▶│  Retriever   │────▶│  Reranker      │
        │ (optional)  │     │ (dense/hybrid│     │  (optional,    │
        └─────────────┘     │  /composed)  │     │  cross-encoder)│
                            └──────────────┘     └────────────────┘
                                                        │
                                                        ▼
                                                ┌────────────────┐
                                                │  gpt-4o-mini   │
                                                │  (grounded)    │
                                                └────────────────┘
                                                        │
                                                        ▼
                                                ┌────────────────┐
                                                │   evaluation   │
                                                │  recall +      │
                                                │  faithfulness  │
                                                │  + correctness │
                                                └────────────────┘
```

## Package layout

```
src/rag_harness/
├── config.py               # Pydantic settings loaded from .env
├── models.py               # Chunk, GoldenCase, EvalResult, EvalSummary
├── logging_setup.py        # Centralised log format + level control
├── ingest/
│   ├── loader.py           # Clone the pinned K8s docs snapshot
│   ├── chunker.py          # Heading-aware markdown chunking
│   ├── embedder.py         # Batched OpenAI embeddings (with cache)
│   ├── embedding_cache.py  # SQLite cache keyed by SHA-256(model+text)
│   └── indexer.py          # Idempotent upsert into ChromaDB
├── retrieval/
│   ├── base.py             # Abstract Retriever interface
│   ├── dense.py            # ChromaDB cosine similarity
│   ├── bm25_store.py       # In-memory BM25Okapi index
│   ├── hybrid.py           # Dense + BM25 with Reciprocal Rank Fusion
│   ├── reranker.py         # Cross-encoder reranking (optional [rerank])
│   ├── hyde.py             # Hypothetical Document Embeddings
│   └── factory.py          # build_retriever(strategy) composes them
├── generation/
│   ├── generator.py        # Context-only prompt, gpt-4o-mini, temp=0
│   ├── critic.py           # Relevance critic (CRAG-style batch scoring)
│   └── corrective.py       # corrective_generate() — retrieve → critique → route
├── evaluation/
│   ├── metrics.py          # context_recall, faithfulness, correctness
│   └── runner.py           # Golden set loader + gate + JSON/CSV export
└── api/
    ├── server.py           # FastAPI: POST /query, GET /health
    └── cli.py              # argparse: ingest, query, eval subcommands
```

## Evaluation

- **Golden set** — 30 hand-verified cases in `evals/golden/`, organised by
  topic (workloads, networking, storage, scheduling, cluster). Version
  controlled and reviewed like code; never auto-generated without review.
- **Reliability gate** — the eval suite fails when any of the three metrics
  drops below its threshold. Treated as a build failure, not a warning.
- **LLM-as-judge** — faithfulness and correctness use `gpt-4o-mini` at
  `temperature=0`; context recall is deterministic set intersection.
- **Nightly runs** — the eval workflow runs on schedule to control cost;
  it is not wired into per-PR CI. Results can be exported to JSON or CSV
  for regression tracking.
- **Integration tests** — end-to-end tests in `tests/integration/` exercise
  the real chunker, indexer, and retriever against a small fixture corpus.
  Run with `pytest -m integration`.

## Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Required for embeddings and generation | — |
| `EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `GENERATION_MODEL` | OpenAI chat model | `gpt-4o-mini` |
| `RETRIEVAL_STRATEGY` | Retrieval strategy (see table above) | `dense` |
| `RETRIEVAL_TOP_K` | Chunks returned per query | `5` |
| `HYBRID_RRF_K` | RRF fusion constant | `60` |
| `CHROMA_DB_PATH` | Vector store location | `./chroma_db` |
| `EMBEDDING_CACHE_PATH` | Cache database location | `./embedding_cache.db` |
| `LOG_LEVEL` | Root logger level | `INFO` |

## Design decisions

All non-trivial decisions are captured as Architecture Decision Records in
[`docs/adr/`](docs/adr/):

- [ADR-0001](docs/adr/ADR-0001-chromadb.md) — ChromaDB as vector store
- [ADR-0002](docs/adr/ADR-0002-pinned-corpus.md) — Pinned corpus commit
- [ADR-0003](docs/adr/ADR-0003-heading-based-chunking.md) — Heading-based chunking
- [ADR-0004](docs/adr/ADR-0004-llm-as-judge-evaluation.md) — LLM-as-judge evaluation
- [ADR-0005](docs/adr/ADR-0005-embedding-cache.md) — SQLite embedding cache
- [ADR-0006](docs/adr/ADR-0006-hybrid-retrieval-and-reranking.md) — Hybrid retrieval, reranking, HyDE
- [ADR-0007](docs/adr/ADR-0007-corrective-rag.md) — Corrective RAG critic-and-retry loop

## Research foundations

The retrieval and evaluation design draws on:

- **RAGAS** — reference-free RAG evaluation metrics
- **Cormack et al. 2009** — Reciprocal Rank Fusion for combining rankers
- **HyDE (Gao et al. 2022)** — Precise Zero-Shot Dense Retrieval without Relevance Labels
- **MS MARCO cross-encoders** — retrieve-then-rerank two-stage pattern
- **CRAG (Yan et al. 2024)** — critic-and-retry loop

## Development

```bash
make check         # ruff lint + ruff format + mypy strict + pytest
make test          # tests only
make lint          # lint only
```

Trunk-based development, conventional commits, all changes gated behind
`make check`. Golden set changes are reviewed like code.

## Attribution

Kubernetes documentation is © The Kubernetes Authors, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See
[NOTICE](NOTICE). Project source code is MIT licensed.
