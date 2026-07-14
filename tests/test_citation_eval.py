"""Tests for the citation-accuracy probe's pure parts."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.evaluation.citation_eval import (
    CitationEvalResult,
    aggregate,
    run_citation_eval,
    sentences_with_markers,
)
from rag_harness.models import Chunk, GoldenCase


def test_sentences_with_markers_splits_and_extracts() -> None:
    answer = "A Pod is a unit [1]. It wraps containers [2][3]. No citation here."
    parsed = sentences_with_markers(answer)
    assert len(parsed) == 3
    assert parsed[0] == ("A Pod is a unit [1].", [1])
    assert parsed[1] == ("It wraps containers [2][3].", [2, 3])
    assert parsed[2][1] == []  # uncited sentence


def test_sentences_with_markers_empty_answer() -> None:
    assert sentences_with_markers("   ") == []


def test_result_coverage_and_accuracy() -> None:
    r = CitationEvalResult(
        n_cases=5, n_sentences=10, n_cited_sentences=8, n_citations=12, n_supported=9
    )
    assert r.coverage == pytest.approx(0.8)
    assert r.accuracy == pytest.approx(0.75)


def test_result_zero_guards() -> None:
    r = CitationEvalResult(0, 0, 0, 0, 0)
    assert r.coverage == 0.0
    assert r.accuracy == 0.0


def test_aggregate_sums_per_case() -> None:
    r = aggregate([(4, 3, 5, 4), (6, 5, 7, 6)])
    assert r.n_cases == 2
    assert r.n_sentences == 10
    assert r.n_cited_sentences == 8
    assert r.n_citations == 12
    assert r.n_supported == 10


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c::{i}",
        text=f"text {i}",
        source_file=f"doc-{i}.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=i,
    )


@pytest.mark.asyncio
async def test_run_citation_eval_scores_supported_citations() -> None:
    case = GoldenCase(id="c1", question="q?", reference_answer="ref", relevant_doc_ids=[])
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk(0), _chunk(1)])

    async def fake_generate(question: str, chunks: list) -> str:
        return "Claim one [1]. Claim two [2]. Uncited claim."

    async def fake_faith(question: str, sentence: str, chunks: list) -> float:
        # passage 1 supports its sentence, passage 2 does not
        return 0.9 if "one" in sentence else 0.2

    with (
        patch("rag_harness.evaluation.citation_eval.generate_async", side_effect=fake_generate),
        patch("rag_harness.evaluation.citation_eval.faithfulness_async", side_effect=fake_faith),
    ):
        result = await run_citation_eval([case], retriever)

    assert result.n_sentences == 3
    assert result.n_cited_sentences == 2
    assert result.n_citations == 2
    assert result.n_supported == 1  # only the first citation is supported
    assert result.accuracy == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_citation_eval_skips_out_of_range_markers() -> None:
    case = GoldenCase(id="c1", question="q?", reference_answer="ref", relevant_doc_ids=[])
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk(0)])  # only 1 chunk

    async def fake_generate(question: str, chunks: list) -> str:
        return "Cites a missing passage [5]."

    with (
        patch("rag_harness.evaluation.citation_eval.generate_async", side_effect=fake_generate),
        patch(
            "rag_harness.evaluation.citation_eval.faithfulness_async",
            new_callable=AsyncMock,
            return_value=1.0,
        ),
    ):
        result = await run_citation_eval([case], retriever)

    # [5] has no matching chunk -> not counted as a citation
    assert result.n_citations == 0
