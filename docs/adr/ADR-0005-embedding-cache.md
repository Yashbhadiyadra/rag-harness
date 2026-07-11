# ADR-0005 - SQLite embedding cache

**Status:** Accepted  
**Date:** 2026-07-01  
**Decided by:** Owner, 2026-07-01

## Context

During development and CI, the corpus is re-ingested frequently. Every ingest
call sends each chunk's text to the OpenAI Embeddings API, incurring latency
and cost. The corpus is pinned to an immutable git commit (ADR-0002), so chunk
texts are stable between runs: the same text always produces the same vector for
a given model.

The cache must be:
- **Model-aware**: changing the embedding model must invalidate old entries.
- **Persistent**: survive process restarts so a warm cache carries across runs.
- **Dependency-free**: no new runtime packages; keeps the install simple.

## Decision

Use a local **SQLite database** as the cache store. Each row maps a SHA-256 digest
of `"{model}\n{text}"` to the JSON-serialised embedding vector. The key encodes
both the model name and the chunk text, so switching models produces cache misses
automatically without any explicit invalidation logic.

`EmbeddingCache` is passed into `embed_chunks()` as an optional argument so the
function remains pure and fully testable without a database.

## Alternatives considered

| Option | Why rejected |
|---|---|
| JSON flat file | Requires loading the entire file into memory and re-serialising on every write; slow for thousands of entries |
| `shelve` (stdlib) | Platform-specific file format (`.db`, `.dir`, `.bak` depending on OS); unreliable across environments |
| `diskcache` (third-party) | Would work well, but adds a dependency; SQLite from stdlib is sufficient |
| In-memory dict | Does not survive restarts; re-embeds on every new process |

## Consequences

- Re-ingesting an unchanged corpus is free after the first run: zero OpenAI API
  calls for chunks already in the cache.
- The cache file (`embedding_cache.db`) is git-ignored and lives alongside
  `chroma_db/`. Its path is configurable via `EMBEDDING_CACHE_PATH` in `.env`.
- The cache is **not** invalidated automatically if the corpus changes. Because the
  key includes the chunk text, new or changed chunks simply produce cache misses and
  are embedded fresh; removed chunks leave stale rows that are harmless.
- For Project 2 (stale-embedding detection), this cache is a useful side artefact:
  it provides a fast lookup of "was this text already embedded with this model?"
  without hitting the API.
