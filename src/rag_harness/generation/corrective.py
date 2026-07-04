"""Corrective RAG — retrieve, critique, route, and optionally retry.

Implements the corrective loop from Yan et al. 2024. The passive
retrieve-then-generate pipeline becomes an active one that judges its own
retrieval quality:

    query
      → retrieve
      → critic scores every chunk
      → route on the aggregate category:
          CORRECT    — filter out any chunk below the incorrect threshold,
                       generate the answer using the survivors
          AMBIGUOUS  — same as Correct but with a smaller surviving set
          INCORRECT  — reformulate the query, retrieve again (up to
                       corrective_max_retries), re-critique; on final
                       failure return a "not enough information" refusal

The refusal message is identical to the generator's own fallback so downstream
evaluation sees a consistent signal. Query reformulation is a single lightweight
`gpt-4o-mini` call; the reformulation prompt asks for a rephrase that surfaces
different keywords, not a semantic change of intent.
"""

import logging
from dataclasses import dataclass, field

from openai import OpenAI

from rag_harness.config import settings
from rag_harness.generation.critic import Category, RelevanceCritic
from rag_harness.generation.generator import generate
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage
from rag_harness.retrieval.base import Retriever

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = (
    "I do not have enough information in the provided context to answer this question."
)

_REFORMULATE_SYSTEM_PROMPT = """\
You are a search query rewriter. The user asked a question but the first \
retrieval attempt returned nothing relevant. Rewrite the question so it \
surfaces different keywords, terminology, or synonyms. Keep the semantic \
intent unchanged. Output ONLY the rewritten question — no preface, no \
explanation, no punctuation beyond a question mark.
"""


@dataclass
class CorrectiveResult:
    """Answer plus telemetry about the corrective path taken."""

    answer: str
    chunks_used: list[Chunk]
    category: Category
    attempts: int  # 1 = no retry; 2 = one retry; etc.
    scores: list[float] = field(default_factory=list)
    reformulated_query: str | None = None


def _reformulate_query(client: OpenAI, model: str, query: str) -> str:
    """Ask the LLM to rephrase *query* to surface different keywords."""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": _REFORMULATE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        record_usage(TokenUsage.from_openai(model, response))
        rewritten = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("query reformulation failed; retrying with the original query")
        return query
    return rewritten or query


def corrective_generate(
    query: str,
    retriever: Retriever,
    critic: RelevanceCritic | None = None,
    max_retries: int | None = None,
    top_k: int | None = None,
) -> CorrectiveResult:
    """Run the corrective retrieve → critique → route flow.

    Returns a CorrectiveResult carrying the answer, the chunks that made it
    into the final generation, the routing category from the last attempt,
    the number of retrieval attempts, per-chunk scores, and (if used) the
    reformulated query.
    """
    critic = critic or RelevanceCritic()
    retry_cap = max_retries if max_retries is not None else settings.corrective_max_retries

    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.generation_model

    current_query = query
    reformulated: str | None = None
    attempts = 0
    last_category = Category.INCORRECT
    last_scores: list[float] = []

    for attempt in range(retry_cap + 1):
        attempts = attempt + 1
        chunks = retriever.retrieve(current_query, top_k=top_k)
        scores = critic.score_batch(query, chunks)  # score against ORIGINAL query
        category = critic.categorise(scores)

        last_category = category
        last_scores = scores

        logger.info(
            "corrective attempt %d: category=%s top_score=%.2f",
            attempts,
            category.value,
            max(scores) if scores else 0.0,
        )

        if category is Category.CORRECT or category is Category.AMBIGUOUS:
            # Keep chunks above the incorrect threshold; drop the rest
            survivors = [
                c for c, s in zip(chunks, scores, strict=True) if s >= critic._incorrect_threshold
            ]
            answer = generate(query, survivors)
            return CorrectiveResult(
                answer=answer,
                chunks_used=survivors,
                category=category,
                attempts=attempts,
                scores=scores,
                reformulated_query=reformulated,
            )

        # Incorrect — reformulate and retry, unless we've hit the cap
        if attempt < retry_cap:
            reformulated = _reformulate_query(client, model, query)
            current_query = reformulated
            logger.info("reformulated query: %r", reformulated)

    # Exhausted retries with an Incorrect verdict — refuse rather than hallucinate
    return CorrectiveResult(
        answer=NO_INFO_MESSAGE,
        chunks_used=[],
        category=last_category,
        attempts=attempts,
        scores=last_scores,
        reformulated_query=reformulated,
    )
