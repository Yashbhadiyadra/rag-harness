"""Corpus specification - what to ingest and how to pin it.

The harness ships with the Kubernetes docs as its flagship corpus, but the
ingest pipeline is corpus-agnostic: point ``CORPUS_*`` (or pass a
``CorpusSpec``) at any git repo of markdown docs and the same reliability
machinery applies (ADR-0019). The pinned git ref is the corpus checksum -
an immutable content hash that is resolved to a full SHA at ingest and
attached to every chunk as provenance, so a corpus is always reproducible.
"""

from dataclasses import dataclass

from rag_harness.config import settings


@dataclass(frozen=True)
class CorpusSpec:
    """An immutable description of a documentation corpus to ingest."""

    name: str  # short slug, used to isolate the local checkout directory
    repo_url: str  # git remote to clone
    git_ref: str  # pinned commit SHA (preferred) or branch name
    docs_subpath: str  # path within the repo holding the docs
    doc_glob: str = "*.md"  # which files under docs_subpath to ingest

    @property
    def local_dir_name(self) -> str:
        """Per-corpus checkout directory so different corpora do not collide."""
        return f"{self.name}_docs"


def default_corpus() -> CorpusSpec:
    """Build the corpus spec from settings (Kubernetes docs by default)."""
    return CorpusSpec(
        name=settings.corpus_name,
        repo_url=settings.corpus_repo_url,
        git_ref=settings.corpus_git_ref,
        docs_subpath=settings.corpus_docs_subpath,
        doc_glob=settings.corpus_doc_glob,
    )
