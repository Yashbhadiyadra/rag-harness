"""Generate grounded answers from retrieved chunks using a context-only prompt.

Async-first: ``generate_async`` is the real implementation. ``generate`` is a
sync wrapper kept only for callers that have not been migrated yet; it will
be retired once every caller is async (Phase 9 progression).
"""

import asyncio
import logging

from openai import AsyncOpenAI

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical assistant that answers questions about Kubernetes.
Answer using ONLY the information in the provided context passages.
If the context does not contain enough information to answer, say:
"I don't have enough information in the provided context to answer this question."
Do not use any outside knowledge. Be concise and precise.
"""

_client = AsyncOpenAI(api_key=settings.openai_api_key)


def _build_context(chunks: list[Chunk]) -> str:
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.source_file
        sections.append(f"[{i}] {heading}\n{chunk.text}")
    return "\n\n---\n\n".join(sections)


_FALLBACK_ANSWER = (
    "I do not have enough information in the provided context to answer this question."
)


async def generate_async(query: str, chunks: list[Chunk]) -> str:
    """Async implementation — call this directly from async callers."""
    if not chunks:
        return _FALLBACK_ANSWER

    context = _build_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

    response = await _client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,  # deterministic output — important for reproducible eval
    )
    record_usage(TokenUsage.from_openai(settings.generation_model, response))

    answer = response.choices[0].message.content or ""
    logger.debug("generated answer (%d chars) for query: %.60s...", len(answer), query)
    return answer


def generate(query: str, chunks: list[Chunk]) -> str:
    """Sync facade — runs the async implementation in a fresh event loop.

    Kept for the transition period only. New callers should ``await
    generate_async`` directly. This wrapper is removed once every caller
    is async (see Phase 9 commit sequence).
    """
    return asyncio.run(generate_async(query, chunks))
