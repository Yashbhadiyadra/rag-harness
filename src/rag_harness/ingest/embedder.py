"""Embed Chunks in batches using the configured OpenAI embeddings model."""

import logging
from collections.abc import Iterator
from itertools import islice

from openai import OpenAI

from rag_harness.config import settings
from rag_harness.ingest.embedding_cache import EmbeddingCache
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512  # well within OpenAI's 2048-input limit; keeps payloads manageable

_client = OpenAI(api_key=settings.openai_api_key)


def _batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield successive fixed-size batches from a list."""
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def embed_chunks(
    chunks: list[Chunk],
    cache: EmbeddingCache | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """Embed chunks in batches. Returns (chunk, vector) pairs in input order.

    If *cache* is provided, vectors already present in the cache are returned
    without an API call. Only the remaining chunks are sent to OpenAI, and
    their vectors are written back to the cache before returning.
    """
    if not chunks:
        return []

    results: list[tuple[Chunk, list[float]]] = []
    to_embed: list[tuple[int, Chunk]] = []  # (original index, chunk) for cache misses

    # Cache lookup pass
    for i, chunk in enumerate(chunks):
        if cache is not None:
            key = EmbeddingCache.make_key(settings.embedding_model, chunk.text)
            cached = cache.get(key)
            if cached is not None:
                results.append((chunk, cached))
                continue
        to_embed.append((i, chunk))

    hit_count = len(chunks) - len(to_embed)
    if hit_count:
        logger.info("embedding cache: %d hits, %d misses", hit_count, len(to_embed))

    # API call for cache misses, in batches
    miss_results: list[tuple[Chunk, list[float]]] = []
    miss_chunks = [c for _, c in to_embed]

    for batch in _batched(miss_chunks, _BATCH_SIZE):
        texts = [c.text for c in batch]
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
        for chunk, embedding_obj in zip(batch, response.data):
            vector = embedding_obj.embedding
            miss_results.append((chunk, vector))
            if cache is not None:
                key = EmbeddingCache.make_key(settings.embedding_model, chunk.text)
                cache.set(key, vector)
        logger.debug("embedded batch of %d chunks via API", len(batch))

    results.extend(miss_results)
    logger.info("embedded %d chunks total (%d from cache)", len(results), hit_count)
    return results
