"""Tests for the query-decomposition retriever."""

from unittest.mock import AsyncMock, patch

import pytest

from rag_harness.models import Chunk
from rag_harness.retrieval.decomposition import DecompositionRetriever, parse_subqueries


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        text=f"text {cid}",
        source_file=f"{cid}.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
    )


def test_parse_subqueries_multi_line() -> None:
    raw = "What is a StatefulSet?\nHow does Pod naming differ from a Deployment?"
    assert parse_subqueries(raw, "orig") == [
        "What is a StatefulSet?",
        "How does Pod naming differ from a Deployment?",
    ]


def test_parse_subqueries_strips_numbering_and_bullets() -> None:
    raw = "1. first query\n- second query\n* third query"
    assert parse_subqueries(raw, "orig") == ["first query", "second query", "third query"]


def test_parse_subqueries_empty_falls_back_to_original() -> None:
    assert parse_subqueries("", "the original") == ["the original"]
    assert parse_subqueries("   \n  \n", "the original") == ["the original"]


def test_parse_subqueries_caps_at_four() -> None:
    raw = "\n".join(f"q{i}" for i in range(10))
    assert len(parse_subqueries(raw, "orig")) == 4


@pytest.mark.asyncio
async def test_single_subquery_delegates_to_base() -> None:
    base = AsyncMock()
    base.retrieve_async = AsyncMock(return_value=[_chunk("a"), _chunk("b")])
    retriever = DecompositionRetriever(base_retriever=base)

    with patch.object(retriever, "_decompose", AsyncMock(return_value=["just one"])):
        result = await retriever.retrieve_async("just one", top_k=2)

    base.retrieve_async.assert_awaited_once_with("just one", top_k=2)
    assert [c.id for c in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_multi_subquery_fuses_and_slices() -> None:
    base = AsyncMock()
    # sub-query 1 finds a,b; sub-query 2 finds b,c - b appears in both (rank-fused up)
    base.retrieve_async = AsyncMock(
        side_effect=[[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]]
    )
    retriever = DecompositionRetriever(base_retriever=base)

    with patch.object(retriever, "_decompose", AsyncMock(return_value=["q1", "q2"])):
        result = await retriever.retrieve_async("multi part", top_k=2)

    assert base.retrieve_async.await_count == 2
    ids = [c.id for c in result]
    assert len(ids) == 2  # sliced to top_k
    assert ids[0] == "b"  # b is retrieved by both sub-queries -> fuses to the top
    assert set(ids) <= {"a", "b", "c"}


@pytest.mark.asyncio
async def test_decompose_falls_back_to_raw_query_on_error() -> None:
    base = AsyncMock()
    base.retrieve_async = AsyncMock(return_value=[_chunk("a")])
    retriever = DecompositionRetriever(base_retriever=base)

    # force the LLM call to raise; _decompose must return the raw query
    retriever._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    subqueries = await retriever._decompose("original question")
    assert subqueries == ["original question"]
