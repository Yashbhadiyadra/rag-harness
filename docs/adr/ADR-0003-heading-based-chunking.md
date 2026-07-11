# ADR-0003 - Chunk markdown on heading boundaries, not token count

**Status:** Accepted  
**Date:** 2026-06-30  
**Decided by:** Owner, 2026-06-30

## Context
The K8s docs are structured markdown. We need a chunking strategy that produces
semantically coherent units suitable for embedding and retrieval. Two common approaches:

- **Token-count chunking**: split every N tokens, optionally with overlap.
- **Heading-based chunking**: split at heading boundaries (`#`, `##`, `###`),
  treating each section as one chunk.

## Decision
Split on heading boundaries (up to `###`). Each chunk is the text under one heading,
up to a 512-token guard. Chunks that exceed 512 tokens are skipped with a warning
rather than truncated.

Track the full heading hierarchy above each chunk as `heading_path`
(e.g. `["Security", "RBAC", "Role Binding"]`) and store it as provenance metadata.

## Reasons
- **Semantic coherence**: a heading boundary is a semantic boundary. Text under
  `### Role Binding` is a self-contained concept. A mid-paragraph token split
  is not.
- **Retrieval quality**: embedding a coherent section produces a more meaningful
  vector than embedding an arbitrary token window.
- **Heading path as context**: the path tells the retriever (and the reader) where
  in the document the chunk lives, without needing to re-read the full file.

## Tradeoffs
- **Oversized sections are dropped, not truncated.** Truncation would corrupt
  provenance: the chunk would claim to represent a full section but would not.
  Dropping with a warning is honest. Affected sections are typically auto-generated
  reference tables, not narrative prose.
- **Heading depth capped at `###`.** Deeper headings (`####` etc.) are rare in the
  K8s docs and typically introduce very small subsections; including them would
  produce many tiny chunks. They are merged into their `###` parent.
- **No overlap.** Overlap is a token-count-chunking technique. With heading-based
  chunks, the heading path provides the context that overlap would otherwise supply.

## Alternatives considered
- **Token-count with overlap (256 tokens, 64 overlap)**: discarded because it
  breaks semantic units and makes provenance ambiguous: a chunk does not belong
  to a single section.
- **Sentence-level chunking**: too granular; individual sentences rarely carry
  enough context to produce a useful retrieval hit.
