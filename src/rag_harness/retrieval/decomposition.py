"""Query decomposition - split a multi-part question, retrieve, fuse.

A single dense query struggles with multi-hop questions ("What is a
StatefulSet and how does its Pod naming differ from a Deployment?") because
one embedding cannot sit near both answers at once. Decomposition asks an
LLM to break the question into focused sub-queries, retrieves for each with
a base retriever, and fuses the ranked lists with Reciprocal Rank Fusion -
so evidence for every part of the question can surface.

Like HyDE, this is a query-time transform that composes with any base
retriever. A single-intent question decomposes to itself, so the strategy
is safe to use everywhere; it only helps when there is more than one thing
to retrieve.
"""

import logging

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage
from rag_harness.openai_client import build_async_client
from rag_harness.retrieval.base import Retriever
from rag_harness.retrieval.hybrid import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM_PROMPT = (
    "You split a user's question into the minimal set of standalone search "
    "queries needed to answer it. If the question asks only one thing, return "
    "it unchanged as the single query. If it asks several things, return one "
    "focused query per thing. Output ONLY the queries, one per line, no "
    "numbering, no preface. Never output more than 4."
)

_MAX_SUBQUERIES = 4


def parse_subqueries(raw: str, original: str) -> list[str]:
    """Parse the LLM's line-separated sub-queries; fall back to the original.

    Empty output or a single line means no useful decomposition - return the
    original query alone. Caps the count so a misbehaving model cannot fan
    retrieval out without bound.
    """
    lines = [line.strip().lstrip("-*0123456789. ").strip() for line in raw.splitlines()]
    subqueries = [line for line in lines if line]
    if not subqueries:
        return [original]
    return subqueries[:_MAX_SUBQUERIES]


class DecompositionRetriever(Retriever):
    """Decomposes a query into sub-queries, retrieves each, fuses with RRF."""

    def __init__(
        self,
        base_retriever: Retriever,
        model: str | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self._base = base_retriever
        self._model = model or settings.generation_model
        self._k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
        self._client = build_async_client()

    async def _decompose(self, query: str) -> list[str]:
        """Ask the LLM to split *query*; fall back to the raw query on failure."""
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            )
        except Exception:
            logger.warning("decomposition failed; falling back to raw query")
            return [query]
        record_usage(TokenUsage.from_openai(self._model, response))
        return parse_subqueries(response.choices[0].message.content or "", query)

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve for each sub-query and fuse the ranked lists with RRF.

        Each sub-query retrieves the full top_k so evidence for a minor part
        of the question is not starved; RRF then re-ranks the union and the
        caller's top_k slices the fused result.
        """
        subqueries = await self._decompose(query)
        logger.debug("decomposed %r into %d sub-queries", query[:60], len(subqueries))
        if len(subqueries) == 1:
            return await self._base.retrieve_async(subqueries[0], top_k=top_k)

        ranked_lists = [await self._base.retrieve_async(sq, top_k=top_k) for sq in subqueries]
        fused = reciprocal_rank_fusion(ranked_lists, k=self._k)
        return fused[:top_k] if top_k is not None else fused
