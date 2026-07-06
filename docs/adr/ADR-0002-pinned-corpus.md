# ADR-0002 — Pin the K8s docs corpus to a specific git commit

**Status:** Accepted  
**Date:** 2026-06-30  
**Decided by:** Owner, 2026-06-30

## Context
The Kubernetes documentation changes with every release. Reproducible evaluation
requires a stable corpus — if the docs change between runs, metric changes could
be caused by content drift rather than pipeline changes.

## Decision
Ingest from a single pinned git commit SHA, stored in `Settings.k8s_git_commit`.
Retain a bounded window of release history as the change stream for Project 2.

## Reasons
- Reproducible: two ingest runs from the same commit produce identical chunks
- Required by Project 2: stale-embedding detection needs to compare chunk content
  at commit N against commit N+1 — only possible if commit provenance is recorded
- Explicit upgrade path: advancing the pinned commit is a deliberate, reviewable act

## Tradeoffs
- Docs may be outdated relative to the latest K8s release
- Advancing the pin requires re-embedding the changed files (mitigated by caching)
