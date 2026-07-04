"""Hybrid retrieval combining dense semantic search with BM25 sparse search.

Dense retrieval excels at semantic matches ("what maintains pod availability
during upgrades?" → docs about PodDisruptionBudget). BM25 excels at exact-term
matches ("PodDisruptionBudget" → same doc). Fusing them with Reciprocal Rank
Fusion (RRF) gets the best of both without hyperparameter tuning:

    score(d) = Σ_i 1 / (k + rank_i(d))

where rank_i(d) is d's position in the i-th ranker's output list, and k is a
constant (60 is the standard from Cormack et al. 2009). Documents that appear
near the top in both rankers get the highest fused score.

RRF has one important property: it is rank-based, so the two rankers' raw
scores never need to be normalised. This makes it robust across corpus sizes
and query types.
"""

import logging

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.retrieval.base import Retriever
from rag_harness.retrieval.bm25_store import BM25Store
from rag_harness.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    k: int = 60,
) -> list[Chunk]:
    """Fuse multiple ranked lists into a single ranking using RRF.

    Documents are deduplicated by Chunk.id. The k constant defaults to 60,
    the value validated by Cormack et al. 2009 and used across production IR.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            # 1-indexed rank per Cormack et al. 2009; enumerate is 0-indexed
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            chunks_by_id[chunk.id] = chunk

    ranked_ids = sorted(scores.keys(), key=lambda cid: -scores[cid])
    return [chunks_by_id[cid] for cid in ranked_ids]


class HybridRetriever(Retriever):
    """Combines DenseRetriever and BM25Store with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        bm25: BM25Store | None = None,
        rrf_k: int | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        """Construct a hybrid retriever.

        candidate_multiplier controls how many candidates each backing ranker
        returns before fusion. Fusing top-15 dense + top-15 BM25 to produce a
        stable top-5 is the standard pattern — the extra candidates ensure the
        best documents survive the fusion even if only one ranker ranks them
        highly.
        """
        self._dense = dense or DenseRetriever()
        self._bm25 = bm25 or BM25Store()
        self._k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
        self._multiplier = candidate_multiplier

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve with hybrid dense + BM25 fusion."""
        final_k = top_k if top_k is not None else settings.retrieval_top_k
        candidate_k = final_k * self._multiplier

        dense_hits = await self._dense.retrieve_async(query, top_k=candidate_k)
        # BM25 search is in-memory and fast; keep sync.
        bm25_hits = [chunk for chunk, _score in self._bm25.search(query, top_k=candidate_k)]

        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], k=self._k)
        logger.debug(
            "hybrid retrieval: %d dense + %d bm25 → %d fused → returning %d",
            len(dense_hits),
            len(bm25_hits),
            len(fused),
            final_k,
        )
        return fused[:final_k]
