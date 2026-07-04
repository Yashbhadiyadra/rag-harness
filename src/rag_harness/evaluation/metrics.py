"""Evaluation metrics: deterministic context recall and LLM-as-judge faithfulness/correctness."""

import logging
from pathlib import Path

from openai import OpenAI

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.observability.llm_cache import LLMResponseCache
from rag_harness.observability.usage import TokenUsage, record_usage

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)

# Cache handle is lazy — opened on first use only when llm_cache_enabled=true.
_cache: LLMResponseCache | None = None


def _get_cache() -> LLMResponseCache | None:
    """Return the shared LLM cache handle, opening it if enabled and not yet open."""
    global _cache
    if not settings.llm_cache_enabled:
        return None
    if _cache is None:
        _cache = LLMResponseCache(Path(settings.llm_cache_path))
    return _cache


_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Given a question, a context (a set of passages), \
and an answer, decide whether every claim in the answer is supported by the context.

Return a score between 0.0 and 1.0:
- 1.0 means every claim in the answer is directly supported by the context.
- 0.0 means the answer contains claims not found in the context at all.
- Partial scores reflect the fraction of claims that are supported.

Respond with ONLY a decimal number, nothing else. Example: 0.85
"""

_CORRECTNESS_PROMPT = """\
You are an evaluation judge. Given a question, a reference answer, and a generated answer, \
decide how correct the generated answer is compared to the reference.

Return a score between 0.0 and 1.0:
- 1.0 means the generated answer is fully correct and captures all key points.
- 0.0 means the generated answer is wrong or completely unrelated.
- Partial scores reflect partial correctness.

Respond with ONLY a decimal number, nothing else. Example: 0.75
"""

_ANSWER_RELEVANCY_PROMPT = """\
You are an evaluation judge. Given a question and an answer, decide whether the \
answer is on-topic and directly addresses the question — regardless of whether \
the answer is factually correct.

This measures topicality only, not accuracy. An answer can be highly relevant \
(directly on-topic) and still be wrong; that combination is the most dangerous \
failure mode.

Return a score between 0.0 and 1.0:
- 1.0 means the answer is fully on-topic and squarely addresses the question.
- 0.5 means the answer partially addresses the question or is only tangentially related.
- 0.0 means the answer is off-topic or does not address the question at all.

Respond with ONLY a decimal number, nothing else. Example: 0.9
"""

_CONTEXT_PRECISION_PROMPT = """\
You are an evaluation judge. Given a question, a reference answer, and a list of \
retrieved passages, decide what fraction of the passages materially contain \
information needed to produce the reference answer.

This measures retrieval precision: were the retrieved chunks actually useful, or \
was the retriever fetching noise alongside signal?

Return a score between 0.0 and 1.0:
- 1.0 means every retrieved passage contains information supporting the reference answer.
- 0.5 means half the passages are useful; the other half are noise.
- 0.0 means none of the passages contain information supporting the reference answer.

Respond with ONLY a decimal number, nothing else. Example: 0.6
"""


def _llm_score(system_prompt: str, user_message: str) -> float:
    """Call the generation model as a judge and parse its 0–1 score response.

    When the LLM cache is enabled (see ``settings.llm_cache_enabled``), the
    ``(model, system_prompt, user_message)`` triple is looked up first. Cache
    hits skip the API call entirely — no TokenUsage is recorded because no
    tokens were consumed. On a miss, the raw response is stored so subsequent
    identical calls are free.
    """
    cache = _get_cache()
    cache_key: str | None = None
    if cache is not None:
        cache_key = LLMResponseCache.make_key(
            settings.generation_model, system_prompt, user_message
        )
        cached_raw = cache.get(cache_key)
        if cached_raw is not None:
            return _parse_score(cached_raw)

    response = _client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    record_usage(TokenUsage.from_openai(settings.generation_model, response))
    raw = (response.choices[0].message.content or "0").strip()
    if cache is not None and cache_key is not None:
        cache.set(cache_key, raw)
    return _parse_score(raw)


def _parse_score(raw: str) -> float:
    """Parse a raw judge response into a clamped [0.0, 1.0] float."""
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        logger.warning("LLM judge returned non-numeric score: %r — defaulting to 0.0", raw)
        return 0.0


def context_recall(retrieved_chunks: list[Chunk], relevant_doc_ids: list[str]) -> float:
    """Fraction of relevant doc IDs that appear in the retrieved chunks.

    This is a deterministic metric — no LLM call needed. It measures whether
    the retrieval stage fetched the right source files.
    """
    if not relevant_doc_ids:
        return 1.0
    retrieved_files = {c.source_file for c in retrieved_chunks}
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_files)
    return hits / len(relevant_doc_ids)


def faithfulness(question: str, answer: str, retrieved_chunks: list[Chunk]) -> float:
    """Score whether every claim in the answer is grounded in the retrieved context."""
    context = "\n\n".join(c.text for c in retrieved_chunks)
    user_message = f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}"
    return _llm_score(_FAITHFULNESS_PROMPT, user_message)


def correctness(question: str, answer: str, reference_answer: str) -> float:
    """Score how correct the generated answer is relative to the reference answer."""
    user_message = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"Generated answer: {answer}"
    )
    return _llm_score(_CORRECTNESS_PROMPT, user_message)


def answer_relevancy(question: str, answer: str) -> float:
    """Score whether the answer is on-topic for the question (regardless of correctness).

    Complements ``correctness``: an answer can score high on relevancy and low on
    correctness — that combination is confident-sounding hallucination and is
    highlighted as its own failure category in Phase 8's ablation output.
    """
    if not answer.strip():
        return 0.0
    user_message = f"Question: {question}\n\nAnswer: {answer}"
    return _llm_score(_ANSWER_RELEVANCY_PROMPT, user_message)


def context_precision(question: str, retrieved_chunks: list[Chunk], reference_answer: str) -> float:
    """Score the fraction of retrieved chunks that materially support the reference answer.

    Complements ``context_recall``: recall answers "did we get all the right
    chunks?", precision answers "of what we retrieved, how much was useful?".
    """
    if not retrieved_chunks:
        return 0.0
    passages = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(retrieved_chunks))
    user_message = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"Retrieved passages:\n{passages}"
    )
    return _llm_score(_CONTEXT_PRECISION_PROMPT, user_message)
