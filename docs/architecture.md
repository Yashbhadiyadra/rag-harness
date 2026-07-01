# Architecture

## Overview

A reliability-first RAG pipeline over the Kubernetes documentation. The goal is to
**measure** answer quality across three independent failure modes, not merely to answer
questions.

## Data flow

```
K8s docs repo (pinned git commit)
        │
        ▼
    [ingest]
    Load markdown → clean → chunk (preserving heading hierarchy)
    → embed (text-embedding-3-small) → store in ChromaDB with provenance
        │
        ▼
    [retrieval]
    Query → embed → top-k cosine similarity search in ChromaDB
        │
        ▼
    [generation]
    Query + retrieved chunks → gpt-4o-mini → grounded answer
        │
        ▼
    [evaluation]
    Answer + chunks + golden case → context_recall, faithfulness, correctness
```

## Modules

| Module | Responsibility |
|---|---|
| `ingest` | Load K8s docs from a pinned git snapshot, clean, chunk, embed, index |
| `retrieval` | Query → candidate chunks (dense; hybrid-ready interface) |
| `generation` | Retrieved context + query → grounded answer |
| `evaluation` | Score outputs: context recall, faithfulness, correctness |
| `api` | FastAPI service + CLI |

## Key design decisions

See `docs/adr/` for the full decision records.

- **ADR-0001**: ChromaDB as vector store
- **ADR-0002**: Pinned corpus commit

## Forward compatibility

This is Project 1 of a three-part reliability portfolio:

- **Project 2 (stale-embedding detection)** consumes ingest history — every chunk records
  source file path and git commit SHA as provenance.
- **Project 3 (agent trajectory evaluator)** reuses the evaluation layer — metrics and
  golden-set format are kept generic, not hard-wired to single-turn RAG.
- **v2 (corrective RAG)** — generation and evaluation are kept modular to support a
  future critic-and-retry loop without a full rewrite.
