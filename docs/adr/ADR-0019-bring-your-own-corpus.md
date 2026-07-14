# ADR-0019: Bring-your-own-corpus ingestion

Date: 2026-07-14
Status: Accepted

## Context

The ingest pipeline was hardwired to the Kubernetes docs through
`k8s_repo_url` / `k8s_git_commit` / `k8s_docs_subpath` settings. That made
the project "a K8s RAG harness" rather than what the strategy research
argues it should be: a reliability harness you point at *your* docs. The
reliability machinery (judge audit, security eval, ablation, golden-set
methodology) is corpus-independent; only ingestion was tied to one corpus.

## Decision

Introduce a `CorpusSpec` (name, repo_url, git_ref, docs_subpath, doc_glob)
and make the loader and ingest pipeline take it, defaulting to the
Kubernetes docs. The corpus is configured through `CORPUS_*` settings, so
pointing the harness at a different markdown docs repo is a config change,
not a code change:

```
CORPUS_NAME=mydocs
CORPUS_REPO_URL=https://github.com/acme/handbook.git
CORPUS_GIT_REF=<pinned commit sha>
CORPUS_DOCS_SUBPATH=docs
CORPUS_DOC_GLOB=*.md
```

Kubernetes stays the flagship default, so nothing breaks for the existing
setup. Each corpus checks out into its own `\<name\>_docs` directory so
different corpora do not collide on disk.

The pinned git ref is the corpus **checksum**: an immutable content hash
that is resolved to a full commit SHA at ingest and attached to every chunk
as provenance (ADR-0002). This preserves the reproducibility guarantee for
any corpus, not just K8s. For a git repo the SHA is the integrity check; a
non-git source would need its own checksum, which is out of scope while only
git repos are supported.

## Consequences

- The harness is now genuinely reusable: any team can evaluate retrieval
  reliability over their own documentation with the same measured gates,
  judge audit, and security probes. This is the "product, not project"
  step from the strategy research.
- Backward compatible: the K8s defaults are unchanged, and the integration
  pipeline test (which drives `chunk_docs` directly) is unaffected.
- Follow-ups, deliberately out of scope here to keep the change focused:
  (1) auto-deriving `chroma_collection` from the corpus name so switching
  corpora cannot mix indexes (today the operator sets `CHROMA_COLLECTION`);
  (2) a per-corpus golden set, since the shipped golden set is K8s-specific;
  (3) non-git sources (local directories, object storage).
