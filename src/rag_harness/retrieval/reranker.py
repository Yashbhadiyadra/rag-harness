"""Cross-encoder reranking on top of any base retriever.

Rationale - the standard two-stage IR pattern:
    1. First stage retrieves broadly (top-50) with a cheap ranker (dense, BM25,
       or hybrid). Recall matters here; precision does not.
    2. Second stage rescores each candidate with a cross-encoder that reads
       (query, document) as a single sequence and outputs a relevance score.
       Precision matters here; the small candidate set keeps compute bounded.

Cross-encoders read the query and the document jointly, so they can catch
fine-grained relevance signals that bi-encoders (like the OpenAI embedding
model) miss. The tradeoff is O(top_k) forward passes per query - a few dozen
milliseconds on CPU with a small model.

Default model - `cross-encoder/ms-marco-MiniLM-L-6-v2` - is the standard
production choice: trained on MS MARCO passage ranking, 6-layer MiniLM,
runs on CPU at ~80-120ms for 20 candidates, licence-clean.
"""

import asyncio
import logging
from typing import cast

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class RerankingRetriever(Retriever):
    """Wraps any Retriever with a cross-encoder rerank stage.

    On query, retrieves top_k * candidate_multiplier from the base retriever,
    scores each (query, chunk) pair with a cross-encoder, and returns the
    top_k highest-scoring chunks.

    The heavy `sentence-transformers` dependency is imported lazily so the
    module itself is safe to import in environments where the extra isn't
    installed. Instantiating RerankingRetriever without the extra raises a
    helpful ImportError.
    """

    def __init__(
        self,
        base_retriever: Retriever,
        model_name: str | None = None,
        candidate_multiplier: int | None = None,
    ) -> None:
        """Build a reranker on top of *base_retriever*.

        Loads the cross-encoder model at construction time; the first call
        also downloads the model weights (~90MB) if not already cached by
        sentence-transformers.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "RerankingRetriever requires the [rerank] extra. Install with: "
                "pip install -e '.[rerank]'"
            ) from e

        self._base = base_retriever
        self._model_name = model_name or settings.rerank_model
        self._multiplier = (
            candidate_multiplier
            if candidate_multiplier is not None
            else settings.rerank_candidate_multiplier
        )
        self._model = CrossEncoder(self._model_name)
        logger.info("cross-encoder reranker loaded: %s", self._model_name)

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve candidates from the base retriever, then rerank to top_k."""
        final_k = top_k if top_k is not None else settings.retrieval_top_k
        candidate_k = final_k * self._multiplier

        candidates = await self._base.retrieve_async(query, top_k=candidate_k)
        if not candidates:
            return []

        pairs = [[query, chunk.text] for chunk in candidates]
        # Cross-encoder .predict() is CPU-bound; wrap in to_thread so we do
        # not block the event loop while other requests are in flight.
        raw_scores = await asyncio.to_thread(self._model.predict, pairs)
        scores = cast(list[float], list(raw_scores))

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: -pair[1],
        )
        logger.debug(
            "rerank: %d candidates → top %d (score range %.3f–%.3f)",
            len(candidates),
            final_k,
            min(scores),
            max(scores),
        )
        return [chunk for chunk, _ in ranked[:final_k]]
