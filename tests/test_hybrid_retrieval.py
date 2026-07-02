"""Unit tests for Reciprocal Rank Fusion and hybrid retrieval composition."""

from unittest.mock import MagicMock

from rag_harness.models import Chunk
from rag_harness.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        text=f"text for {cid}",
        source_file=f"docs/{cid}.md",
        git_commit="abc123",
        doc_version="v1.32",
        chunk_index=0,
        heading_path=[],
    )


# --- reciprocal_rank_fusion ---


def test_rrf_single_list_preserves_order() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([chunks])
    assert [c.id for c in fused] == ["a", "b", "c"]


def test_rrf_deduplicates_across_lists() -> None:
    list1 = [_chunk("a"), _chunk("b")]
    list2 = [_chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([list1, list2])
    ids = [c.id for c in fused]
    assert set(ids) == {"a", "b", "c"}
    assert len(ids) == 3  # no duplicates


def test_rrf_boosts_documents_appearing_in_both_lists() -> None:
    # b appears rank 1 in list1 and rank 0 in list2 → highest fused score
    # a appears only in list1 at rank 0
    # c appears only in list2 at rank 1
    list1 = [_chunk("a"), _chunk("b")]
    list2 = [_chunk("b"), _chunk("c")]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    # b is in both, so it should rank first
    assert fused[0].id == "b"


def test_rrf_respects_k_constant() -> None:
    # With k=0 the scores are 1/rank+1; with large k the scores flatten.
    list1 = [_chunk("a"), _chunk("b")]
    fused_small_k = reciprocal_rank_fusion([list1], k=0)
    fused_large_k = reciprocal_rank_fusion([list1], k=1000)
    # Order should be preserved in both cases for a single list
    assert [c.id for c in fused_small_k] == ["a", "b"]
    assert [c.id for c in fused_large_k] == ["a", "b"]


def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --- HybridRetriever ---


def test_hybrid_retriever_fuses_dense_and_bm25() -> None:
    dense_mock = MagicMock()
    dense_mock.retrieve.return_value = [_chunk("a"), _chunk("b"), _chunk("c")]

    bm25_mock = MagicMock()
    bm25_mock.search.return_value = [
        (_chunk("c"), 1.5),
        (_chunk("a"), 1.0),
        (_chunk("d"), 0.5),
    ]

    retriever = HybridRetriever(dense=dense_mock, bm25=bm25_mock, rrf_k=60)
    results = retriever.retrieve("query", top_k=3)

    ids = [c.id for c in results]
    # a and c appear in both, so they should outrank b (dense-only) and d (bm25-only)
    assert set(ids[:2]) == {"a", "c"}
    assert len(results) == 3


def test_hybrid_retriever_respects_top_k() -> None:
    dense_mock = MagicMock()
    dense_mock.retrieve.return_value = [_chunk(str(i)) for i in range(10)]

    bm25_mock = MagicMock()
    bm25_mock.search.return_value = [(_chunk(str(i)), 1.0) for i in range(10)]

    retriever = HybridRetriever(dense=dense_mock, bm25=bm25_mock)
    results = retriever.retrieve("query", top_k=5)

    assert len(results) == 5


def test_hybrid_retriever_requests_candidate_multiplier_from_each_ranker() -> None:
    dense_mock = MagicMock()
    dense_mock.retrieve.return_value = []
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = []

    retriever = HybridRetriever(dense=dense_mock, bm25=bm25_mock, candidate_multiplier=3)
    retriever.retrieve("query", top_k=5)

    # Each backing ranker should receive 5*3=15
    dense_mock.retrieve.assert_called_once_with("query", top_k=15)
    bm25_mock.search.assert_called_once_with("query", top_k=15)
