import logging
from pathlib import Path

from git import InvalidGitRepositoryError, Repo

from rag_harness.config import settings

logger = logging.getLogger(__name__)


def _repo_at_correct_commit(repo_path: Path, target_commit: str) -> bool:
    """Return True if the repo exists and HEAD matches the target commit/ref."""
    try:
        repo = Repo(repo_path)
        current = repo.head.commit.hexsha
        # Accept if current SHA starts with target (handles short SHAs) or ref matches
        return current.startswith(target_commit) or target_commit.startswith(current)
    except (InvalidGitRepositoryError, ValueError):
        return False


def ensure_repo(repo_path: Path) -> Repo:
    """Clone the K8s docs repo if absent, then checkout the pinned commit."""
    if repo_path.exists() and _repo_at_correct_commit(repo_path, settings.k8s_git_commit):
        logger.info("repo already at commit %s, skipping clone", settings.k8s_git_commit)
        return Repo(repo_path)

    if repo_path.exists():
        logger.info("repo exists but at wrong commit — fetching and checking out")
        repo = Repo(repo_path)
        repo.remotes.origin.fetch()
    else:
        logger.info("cloning %s into %s", settings.k8s_repo_url, repo_path)
        repo = Repo.clone_from(
            settings.k8s_repo_url,
            repo_path,
            no_checkout=True,
            depth=1 if settings.k8s_git_commit == "main" else None,
        )

    repo.git.checkout(settings.k8s_git_commit)
    logger.info("checked out %s", settings.k8s_git_commit)
    return repo


def load_doc_paths(repo_path: Path) -> list[Path]:
    """Return all .md file paths under the K8s docs subdirectory."""
    docs_root = repo_path / settings.k8s_docs_subpath
    if not docs_root.exists():
        raise FileNotFoundError(
            f"docs directory not found at {docs_root}. Run ensure_repo() first."
        )
    paths = sorted(docs_root.rglob("*.md"))
    logger.info("found %d markdown files under %s", len(paths), docs_root)
    return paths


def load(repo_path: Path | None = None) -> tuple[list[Path], str]:
    """Top-level entry point: ensure repo is ready, return (doc_paths, resolved_commit_sha).

    Returns the resolved commit SHA so callers can attach it to every Chunk
    as provenance — even when k8s_git_commit is a branch name like 'main'.
    """
    target = repo_path or Path(settings.chroma_db_path).parent / "k8s_docs"
    repo = ensure_repo(target)
    commit_sha = repo.head.commit.hexsha
    paths = load_doc_paths(target)
    return paths, commit_sha
