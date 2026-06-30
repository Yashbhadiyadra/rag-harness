import logging
from collections.abc import Iterator
from itertools import islice

from openai import OpenAI

from rag_harness.config import settings
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512  # well within OpenAI's 2048-input limit; keeps payloads manageable

_client = OpenAI(api_key=settings.openai_api_key)


def _batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield successive fixed-size batches from a list."""
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """Embed chunks in batches. Returns (chunk, vector) pairs in the same order as input."""
    if not chunks:
        return []

    results: list[tuple[Chunk, list[float]]] = []

    for batch in _batched(chunks, _BATCH_SIZE):
        texts = [c.text for c in batch]
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        # OpenAI guarantees response order matches input order
        for chunk, embedding_obj in zip(batch, response.data):
            results.append((chunk, embedding_obj.embedding))

        logger.debug("embedded batch of %d chunks", len(batch))

    logger.info("embedded %d chunks total", len(results))
    return results
