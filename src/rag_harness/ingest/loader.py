"""Clone a docs repo at a pinned ref and return its markdown file paths.

Corpus-agnostic (ADR-0019): the repo, ref, subpath, and glob all come from a
``CorpusSpec``, defaulting to the pinned Kubernetes docs.
"""

import logging
from pathlib import Path

from git import InvalidGitRepositoryError, Repo

from rag_harness.config import settings
from rag_harness.ingest.corpus import CorpusSpec, default_corpus

logger = logging.getLogger(__name__)


def _repo_at_correct_commit(repo_path: Path, target_ref: str) -> bool:
    """Return True if the repo exists and HEAD matches the target commit/ref."""
    try:
        repo = Repo(repo_path)
        current = repo.head.commit.hexsha
        # Accept if current SHA starts with target (handles short SHAs) or ref matches
        return current.startswith(target_ref) or target_ref.startswith(current)
    except (InvalidGitRepositoryError, ValueError):
        return False


def ensure_repo(repo_path: Path, corpus: CorpusSpec) -> Repo:
    """Clone the corpus repo if absent, then checkout the pinned ref."""
    if repo_path.exists() and _repo_at_correct_commit(repo_path, corpus.git_ref):
        logger.info("repo already at ref %s, skipping clone", corpus.git_ref)
        return Repo(repo_path)

    if repo_path.exists():
        logger.info("repo exists but at wrong ref - fetching and checking out")
        repo = Repo(repo_path)
        repo.remotes.origin.fetch()
    else:
        logger.info("cloning %s into %s", corpus.repo_url, repo_path)
        repo = Repo.clone_from(
            corpus.repo_url,
            repo_path,
            no_checkout=True,
            depth=1 if corpus.git_ref == "main" else None,
        )

    repo.git.checkout(corpus.git_ref)
    logger.info("checked out %s", corpus.git_ref)
    return repo


def load_doc_paths(repo_path: Path, corpus: CorpusSpec) -> list[Path]:
    """Return all doc file paths under the corpus's docs subdirectory."""
    docs_root = repo_path / corpus.docs_subpath
    if not docs_root.exists():
        raise FileNotFoundError(
            f"docs directory not found at {docs_root}. Run ensure_repo() first."
        )
    paths = sorted(docs_root.rglob(corpus.doc_glob))
    logger.info("found %d files matching %s under %s", len(paths), corpus.doc_glob, docs_root)
    return paths


def load(repo_path: Path | None = None, corpus: CorpusSpec | None = None) -> tuple[list[Path], str]:
    """Top-level entry point: ensure repo is ready, return (doc_paths, resolved_sha).

    Returns the resolved commit SHA so callers can attach it to every Chunk
    as provenance - even when the corpus ref is a branch name like 'main'.
    """
    corpus = corpus or default_corpus()
    target = repo_path or Path(settings.chroma_db_path).parent / corpus.local_dir_name
    repo = ensure_repo(target, corpus)
    commit_sha = repo.head.commit.hexsha
    paths = load_doc_paths(target, corpus)
    return paths, commit_sha
