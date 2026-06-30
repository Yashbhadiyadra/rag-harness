# ADR-0001: Use ChromaDB as the vector store

## Status
Accepted

## Context
We need a vector store to hold embedded chunks and their provenance metadata.
Options considered: ChromaDB, Qdrant, Pinecone, pgvector.

## Decision
Use ChromaDB (local, file-based).

## Reasons
- Zero infrastructure — runs in-process, persists to a local directory
- Supports metadata filtering, which is required for provenance queries (Project 2)
- Easy to inspect and debug during development
- Swappable: retrieval is behind an abstract `Retriever` interface, so replacing
  ChromaDB later does not affect any other module

## Tradeoffs
- Not production-grade for high-concurrency workloads
- No distributed mode — acceptable for a single-node evaluation harness
