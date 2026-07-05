"""Embed Chunks in batches using the configured OpenAI embeddings model."""

import asyncio
import logging
from collections.abc import Iterator
from itertools import islice

from openai import AsyncOpenAI

from rag_harness.config import settings
from rag_harness.ingest.embedding_cache import EmbeddingCache
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512  # well within OpenAI's 2048-input limit; keeps payloads manageable

_client = AsyncOpenAI(api_key=settings.openai_api_key)


def _batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    """Yield successive fixed-size batches from a list."""
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


async def _embed_one_batch(
    batch: list[Chunk],
    cache: EmbeddingCache | None,
    semaphore: asyncio.Semaphore,
) -> list[tuple[Chunk, list[float]]]:
    """Embed a single batch under the concurrency semaphore."""
    async with semaphore:
        texts = [c.text for c in batch]
        response = await _client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
    record_usage(TokenUsage.from_openai(settings.embedding_model, response))
    out: list[tuple[Chunk, list[float]]] = []
    for chunk, embedding_obj in zip(batch, response.data):
        vector = embedding_obj.embedding
        out.append((chunk, vector))
        if cache is not None:
            key = EmbeddingCache.make_key(settings.embedding_model, chunk.text)
            cache.set(key, vector)
    logger.debug("embedded batch of %d chunks via API", len(batch))
    return out


async def embed_chunks_async(
    chunks: list[Chunk],
    cache: EmbeddingCache | None = None,
    concurrency: int | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """Embed chunks concurrently with a bounded semaphore.

    Batches are dispatched via ``asyncio.gather`` so multiple batches are in
    flight against the OpenAI API simultaneously. The semaphore caps the
    concurrency at ``settings.ingest_embed_concurrency`` (default 4) to stay
    within OpenAI rate limits.
    """
    if not chunks:
        return []

    results: list[tuple[Chunk, list[float]]] = []
    to_embed: list[Chunk] = []

    # Cache lookup pass (sync — cache is a local SQLite DB)
    for chunk in chunks:
        if cache is not None:
            key = EmbeddingCache.make_key(settings.embedding_model, chunk.text)
            cached = cache.get(key)
            if cached is not None:
                results.append((chunk, cached))
                continue
        to_embed.append(chunk)

    hit_count = len(chunks) - len(to_embed)
    if hit_count:
        logger.info("embedding cache: %d hits, %d misses", hit_count, len(to_embed))

    if not to_embed:
        return results

    n_conc = concurrency if concurrency is not None else settings.ingest_embed_concurrency
    semaphore = asyncio.Semaphore(n_conc)
    batches = list(_batched(to_embed, _BATCH_SIZE))
    batch_results = await asyncio.gather(*(_embed_one_batch(b, cache, semaphore) for b in batches))
    for batch_out in batch_results:
        results.extend(batch_out)

    logger.info("embedded %d chunks total (%d from cache)", len(results), hit_count)
    return results


def embed_chunks(
    chunks: list[Chunk],
    cache: EmbeddingCache | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """Sync facade — runs the async implementation in a fresh event loop.

    If *cache* is provided, vectors already present in the cache are returned
    without an API call. Only the remaining chunks are sent to OpenAI, and
    their vectors are written back to the cache before returning.
    """
    return asyncio.run(embed_chunks_async(chunks, cache=cache))
