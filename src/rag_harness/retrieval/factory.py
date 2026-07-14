"""Retriever factory - composes retrievers based on a strategy name.

The strategy names are stable identifiers used in configuration, CLI flags,
API requests, and evaluation logs. Adding a new strategy means adding a new
branch here - nothing downstream needs to change.

Strategies:
    dense           - DenseRetriever alone (baseline)
    hybrid          - HybridRetriever (dense + BM25 fused with RRF)
    hybrid-rerank   - HybridRetriever wrapped in RerankingRetriever
    hyde            - HyDERetriever wrapping DenseRetriever
    full            - HyDE ∘ Hybrid ∘ Rerank (the whole pipeline)
    decompose       - DecompositionRetriever wrapping HybridRetriever
"""

import logging

from rag_harness.retrieval.base import Retriever
from rag_harness.retrieval.dense import DenseRetriever
from rag_harness.retrieval.hybrid import HybridRetriever
from rag_harness.retrieval.hyde import HyDERetriever

logger = logging.getLogger(__name__)

VALID_STRATEGIES = {"dense", "hybrid", "hybrid-rerank", "hyde", "full", "decompose"}


def build_retriever(strategy: str) -> Retriever:
    """Return a Retriever composed according to *strategy*.

    Raises ValueError for unknown strategies, and ImportError (from the
    reranker) if a strategy needing sentence-transformers is requested
    without the [rerank] extra installed.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy!r}. Valid choices: {sorted(VALID_STRATEGIES)}"
        )

    logger.info("building retriever with strategy=%s", strategy)

    if strategy == "dense":
        return DenseRetriever()

    if strategy == "hybrid":
        return HybridRetriever()

    if strategy == "hybrid-rerank":
        from rag_harness.retrieval.reranker import RerankingRetriever

        return RerankingRetriever(base_retriever=HybridRetriever())

    if strategy == "hyde":
        return HyDERetriever(base_retriever=DenseRetriever())

    if strategy == "decompose":
        from rag_harness.retrieval.decomposition import DecompositionRetriever

        return DecompositionRetriever(base_retriever=HybridRetriever())

    # "full" - HyDE around Rerank around Hybrid
    from rag_harness.retrieval.reranker import RerankingRetriever

    return HyDERetriever(base_retriever=RerankingRetriever(base_retriever=HybridRetriever()))
