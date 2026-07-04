"""Unit tests for the cross-encoder reranker.

The heavy sentence-transformers dependency is patched everywhere; these tests
run in the default `dev` environment without the `[rerank]` extra installed.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.models import Chunk


def _chunk(cid: str, text: str = "") -> Chunk:
    return Chunk(
        id=cid,
        text=text or f"text for {cid}",
        source_file=f"docs/{cid}.md",
        git_commit="abc123",
        doc_version="v1.32",
        chunk_index=0,
        heading_path=[],
    )


@pytest.fixture()
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a stub sentence_transformers module into sys.modules."""
    fake_module = types.ModuleType("sentence_transformers")
    mock_cross_encoder_cls = MagicMock()
    fake_module.CrossEncoder = mock_cross_encoder_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return mock_cross_encoder_cls


def _mock_base(chunks: list[Chunk]) -> MagicMock:
    base = MagicMock()
    base.retrieve_async = AsyncMock(return_value=chunks)
    return base


def test_reranker_reorders_by_cross_encoder_score(
    fake_sentence_transformers: MagicMock,
) -> None:
    from rag_harness.retrieval.reranker import RerankingRetriever

    base = _mock_base([_chunk("a"), _chunk("b"), _chunk("c")])

    # cross-encoder says c is most relevant, then a, then b
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5, 0.1, 0.9]
    fake_sentence_transformers.return_value = mock_model

    reranker = RerankingRetriever(base_retriever=base, model_name="fake-model")
    results = reranker.retrieve("query", top_k=3)

    assert [c.id for c in results] == ["c", "a", "b"]


def test_reranker_truncates_to_top_k(fake_sentence_transformers: MagicMock) -> None:
    from rag_harness.retrieval.reranker import RerankingRetriever

    base = _mock_base([_chunk(str(i)) for i in range(10)])

    mock_model = MagicMock()
    mock_model.predict.return_value = [float(i) for i in range(10)]  # scores 0..9
    fake_sentence_transformers.return_value = mock_model

    reranker = RerankingRetriever(base_retriever=base)
    results = reranker.retrieve("query", top_k=3)

    # top-3 by score should be 9, 8, 7
    assert [c.id for c in results] == ["9", "8", "7"]


def test_reranker_requests_multiplier_candidates_from_base(
    fake_sentence_transformers: MagicMock,
) -> None:
    from rag_harness.retrieval.reranker import RerankingRetriever

    base = _mock_base([])
    mock_model = MagicMock()
    mock_model.predict.return_value = []
    fake_sentence_transformers.return_value = mock_model

    reranker = RerankingRetriever(base_retriever=base, candidate_multiplier=4)
    reranker.retrieve("query", top_k=5)

    base.retrieve_async.assert_awaited_once_with("query", top_k=20)  # 5 * 4


def test_reranker_handles_empty_candidates(fake_sentence_transformers: MagicMock) -> None:
    from rag_harness.retrieval.reranker import RerankingRetriever

    base = _mock_base([])
    mock_model = MagicMock()
    fake_sentence_transformers.return_value = mock_model

    reranker = RerankingRetriever(base_retriever=base)
    assert reranker.retrieve("query", top_k=5) == []
    # predict should not be called on an empty list
    mock_model.predict.assert_not_called()


def test_reranker_raises_helpful_error_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with patch.dict(sys.modules, {"sentence_transformers": None}):
        # importlib will hit the None entry and raise ImportError
        from rag_harness.retrieval.reranker import RerankingRetriever

        with pytest.raises(ImportError, match=r"\[rerank\] extra"):
            RerankingRetriever(base_retriever=MagicMock())
