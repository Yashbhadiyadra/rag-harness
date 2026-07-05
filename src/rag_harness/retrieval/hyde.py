"""HyDE — Hypothetical Document Embeddings (Gao et al. 2022).

Standard dense retrieval embeds the raw query and searches for chunks with
similar embeddings. This is fragile when the query and answer use different
vocabulary: a question ("How do I stop pods from starting on GPU nodes?")
looks very different from its answer ("Add a taint with effect NoSchedule
to the node...").

HyDE bridges the gap by asking an LLM to first *write* a hypothetical answer,
then embeds the hypothetical answer instead of the raw question. Even when
the LLM hallucinates specific facts, the generated text shares vocabulary,
structure, and topical signal with real answer documents — so its embedding
lands in the right neighbourhood.

Key property: HyDE requires no training or fine-tuning. It composes with any
underlying retriever (dense, hybrid, hybrid+rerank) as a query-time transform.
"""

import logging

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage
from rag_harness.openai_client import build_async_client
from rag_harness.retrieval.base import Retriever

logger = logging.getLogger(__name__)

_HYDE_SYSTEM_PROMPT = (
    "You are a technical writer. Given a question about Kubernetes, write a "
    "short passage (3-5 sentences) that would appear in the official docs as "
    "the answer. Be specific and use accurate terminology. Do not preface, "
    "explain, or apologise — output only the passage."
)


class HyDERetriever(Retriever):
    """Wraps any base retriever with a HyDE query transformation stage."""

    def __init__(
        self,
        base_retriever: Retriever,
        model: str | None = None,
    ) -> None:
        """Build a HyDE retriever around *base_retriever*."""
        self._base = base_retriever
        self._model = model or settings.generation_model
        self._client = build_async_client()

    async def _hypothesise(self, query: str) -> str:
        """Ask the LLM to draft a hypothetical answer passage for *query*."""
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        record_usage(TokenUsage.from_openai(self._model, response))
        content = response.choices[0].message.content or ""
        return content.strip()

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Generate a hypothesis, then retrieve using that as the query.

        If the LLM returns an empty hypothesis (network hiccup, safety refusal,
        etc.) we fall back to the raw query — better a degraded retrieval than
        a failed one.
        """
        try:
            hypothesis = await self._hypothesise(query)
        except Exception:
            logger.warning("HyDE hypothesis generation failed; falling back to raw query")
            hypothesis = ""

        effective_query = hypothesis or query
        logger.debug(
            "HyDE: original query %r → hypothesis (%d chars)",
            query[:60],
            len(hypothesis),
        )
        return await self._base.retrieve_async(effective_query, top_k=top_k)
