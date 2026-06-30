import json
import logging

import chromadb

from rag_harness.config import settings
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 256  # ChromaDB handles large batches fine; this keeps memory predictable


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text embeddings
    )


def _to_metadata(chunk: Chunk) -> dict[str, str]:
    # ChromaDB metadata values must be scalar (str/int/float/bool).
    # heading_path is a list, so we serialize it as JSON.
    return {
        "source_file": chunk.source_file,
        "git_commit": chunk.git_commit,
        "doc_version": chunk.doc_version,
        "chunk_index": str(chunk.chunk_index),
        "heading_path": json.dumps(chunk.heading_path),
    }


def index_chunks(embedded: list[tuple[Chunk, list[float]]]) -> None:
    """Upsert (chunk, vector) pairs into ChromaDB. Idempotent — safe to re-run."""
    if not embedded:
        logger.warning("index_chunks called with empty list — nothing to index")
        return

    collection = _get_collection()

    for batch_start in range(0, len(embedded), _UPSERT_BATCH):
        batch = embedded[batch_start : batch_start + _UPSERT_BATCH]

        collection.upsert(
            ids=[chunk.id for chunk, _ in batch],
            embeddings=[vec for _, vec in batch],  # type: ignore[arg-type]  # chromadb stubs too narrow
            documents=[chunk.text for chunk, _ in batch],
            metadatas=[_to_metadata(chunk) for chunk, _ in batch],
        )
        logger.debug("upserted batch of %d chunks", len(batch))

    logger.info("indexed %d chunks into collection '%s'", len(embedded), settings.chroma_collection)


def collection_count() -> int:
    """Return the number of documents currently in the collection."""
    return _get_collection().count()


def clear_collection() -> None:
    """Delete and recreate the collection. Used in tests and for full re-ingest."""
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    client.delete_collection(settings.chroma_collection)
    client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("cleared collection '%s'", settings.chroma_collection)
