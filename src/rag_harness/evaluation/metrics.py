import logging

from openai import OpenAI

from rag_harness.config import settings
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=settings.openai_api_key)

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


def _llm_score(system_prompt: str, user_message: str) -> float:
    """Call the generation model as a judge and parse its 0–1 score response."""
    response = _client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "0").strip()
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))  # clamp to [0, 1]
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
