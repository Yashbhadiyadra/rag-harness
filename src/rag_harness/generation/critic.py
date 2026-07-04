"""Relevance critic — scores retrieved chunks against a query.

Based on the retrieval evaluator from Corrective RAG (Yan et al. 2024). The
critic reads the query and each retrieved chunk and outputs a 0.0-1.0 relevance
score per chunk. Downstream code categorises the scores into three buckets:

    Correct    — at least one chunk scores above `correct_threshold`
                 (there is a solid answer somewhere in the retrieved set).
    Ambiguous  — no chunk hits Correct but at least one is above
                 `incorrect_threshold` (partial evidence; may still be usable).
    Incorrect  — every chunk is below `incorrect_threshold`
                 (the retriever fetched garbage; do not generate on this).

Design choice — the critic scores all chunks in a single LLM call using
structured JSON output. This is the cost-efficient path: one call per query
instead of N. Calls are made with `temperature=0` for reproducibility.
"""

import json
import logging
from enum import StrEnum
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM_PROMPT = """\
You are a strict retrieval evaluator. Given a user question and a list of \
candidate passages, rate how well each passage answers the question on a \
scale from 0.0 to 1.0.

Scoring rubric:
  1.0  — passage directly and completely answers the question
  0.7  — passage contains most of the answer, or is clearly on-topic
  0.4  — passage is related but does not answer the question
  0.1  — passage is off-topic
  0.0  — passage is irrelevant or unrelated

Return ONLY a JSON object with a single key "scores" whose value is a list of \
floats in the same order as the input passages. No prose, no explanation.
"""


class Category(StrEnum):
    """Coarse routing decision produced from per-chunk scores."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class _CriticResponse(BaseModel):
    """Structured shape of the critic's JSON output."""

    scores: list[float] = Field(default_factory=list)


class RelevanceCritic:
    """Scores retrieved chunks for relevance and routes on the aggregate."""

    def __init__(
        self,
        model: str | None = None,
        correct_threshold: float | None = None,
        incorrect_threshold: float | None = None,
    ) -> None:
        """Build a critic.

        Thresholds default to settings values (0.7 and 0.3). Any chunk above
        `correct_threshold` promotes the whole batch to CORRECT; any batch
        whose max score falls below `incorrect_threshold` is INCORRECT;
        everything in between is AMBIGUOUS.
        """
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.generation_model
        self._correct_threshold = (
            correct_threshold
            if correct_threshold is not None
            else settings.critic_correct_threshold
        )
        self._incorrect_threshold = (
            incorrect_threshold
            if incorrect_threshold is not None
            else settings.critic_incorrect_threshold
        )

    def score_batch(self, query: str, chunks: list[Chunk]) -> list[float]:
        """Return a per-chunk relevance score in [0.0, 1.0].

        On any parsing or network error, returns 0.0 for every chunk — the
        safest default is to treat everything as unknown-relevance and let
        the caller route to INCORRECT. A degraded score is better than
        hallucinating relevance.
        """
        if not chunks:
            return []

        passages = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks))
        user_message = f"Question: {query}\n\nPassages:\n{passages}"

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            record_usage(TokenUsage.from_openai(self._model, response))
            raw = response.choices[0].message.content or "{}"
            parsed = _CriticResponse.model_validate(json.loads(raw))
        except Exception as e:
            logger.warning("critic scoring failed: %s — falling back to zero scores", e)
            return [0.0] * len(chunks)

        scores = [max(0.0, min(1.0, s)) for s in parsed.scores]
        # Truncate or pad so the caller can always zip scores with chunks
        if len(scores) < len(chunks):
            scores = scores + [0.0] * (len(chunks) - len(scores))
        return scores[: len(chunks)]

    def categorise(self, scores: list[float]) -> Category:
        """Reduce a list of per-chunk scores to a single routing decision."""
        if not scores:
            return Category.INCORRECT
        top_score = max(scores)
        if top_score >= self._correct_threshold:
            return Category.CORRECT
        if top_score < self._incorrect_threshold:
            return Category.INCORRECT
        return Category.AMBIGUOUS


CategoryStr = Literal["correct", "ambiguous", "incorrect"]
