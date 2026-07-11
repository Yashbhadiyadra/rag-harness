"""Ingest pipeline: load K8s docs → chunk → embed → index into ChromaDB."""

import asyncio
import logging
from pathlib import Path

from rag_harness.config import settings
from rag_harness.ingest.chunker import chunk_docs
from rag_harness.ingest.embedder import embed_chunks_async
from rag_harness.ingest.embedding_cache import EmbeddingCache
from rag_harness.ingest.indexer import index_chunks
from rag_harness.ingest.loader import load

logger = logging.getLogger(__name__)


async def run_ingest_async(repo_path: Path | None = None) -> None:
    """Async: full ingest pipeline - load → chunk → embed (concurrent) → index."""
    logger.info("starting ingest pipeline")

    doc_paths, commit_sha = load(repo_path)
    doc_version = settings.k8s_git_commit  # human-readable label (branch or tag)

    repo_root = repo_path or Path(settings.chroma_db_path).parent / "k8s_docs"
    chunks = chunk_docs(doc_paths, repo_root, git_commit=commit_sha, doc_version=doc_version)

    logger.info("embedding %d chunks", len(chunks))
    cache = EmbeddingCache(Path(settings.embedding_cache_path))
    try:
        embedded = await embed_chunks_async(chunks, cache=cache)
    finally:
        cache.close()

    logger.info("indexing %d embedded chunks", len(embedded))
    index_chunks(embedded)

    logger.info("ingest complete - %d chunks indexed", len(embedded))


def run_ingest(repo_path: Path | None = None) -> None:
    """Sync facade - runs the async implementation in a fresh event loop."""
    asyncio.run(run_ingest_async(repo_path))
