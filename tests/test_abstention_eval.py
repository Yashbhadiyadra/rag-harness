"""Tests for the abstention (negative-rejection) probe's pure parts."""

from unittest.mock import patch

import pytest

from rag_harness.evaluation.abstention_eval import (
    OUT_OF_CORPUS_QUESTIONS,
    AbstentionResult,
    is_abstention,
    render_markdown,
    run_abstention_probe,
)
from rag_harness.models import GoldenCase

_CASE = GoldenCase(
    id="pods-001",
    question="What is a Pod?",
    reference_answer="A Pod is the smallest deployable unit in Kubernetes.",
    relevant_doc_ids=["content/en/docs/concepts/workloads/pods.md"],
)


def test_is_abstention_matches_refusal_wording() -> None:
    assert is_abstention(
        "I do not have enough information in the provided context to answer this question."
    )
    assert is_abstention("Sorry, I don't have enough information to answer.")
    assert not is_abstention("A Pod is the smallest deployable unit in Kubernetes.")
    # a real answer mentioning 'information' must not count as an abstention
    assert not is_abstention("This information describes how Pods schedule containers.")


def test_abstention_rate() -> None:
    assert AbstentionResult(n_questions=8, n_abstained=8).abstention_rate == 1.0
    assert AbstentionResult(n_questions=8, n_abstained=6).abstention_rate == pytest.approx(0.75)
    assert AbstentionResult(n_questions=0, n_abstained=0).abstention_rate == 1.0


def test_render_markdown_reports_rate() -> None:
    md = render_markdown(AbstentionResult(8, 7), commit="abc1234", timestamp="20260714T000000+0000")
    assert "Abstention" in md
    assert "abc1234" in md
    assert "88%" in md  # 7/8


@pytest.mark.asyncio
async def test_run_abstention_probe_counts_refusals() -> None:
    # a well-behaved model refuses everything it cannot ground
    async def refusing_generate(question: str, chunks: list) -> str:
        return "I do not have enough information in the provided context to answer this question."

    with patch(
        "rag_harness.evaluation.abstention_eval.generate_async", side_effect=refusing_generate
    ):
        result = await run_abstention_probe([_CASE])
    assert result.n_questions == len(OUT_OF_CORPUS_QUESTIONS)
    assert result.abstention_rate == 1.0


@pytest.mark.asyncio
async def test_run_abstention_probe_flags_hallucination() -> None:
    # a model that always answers never abstains -> rate 0
    async def hallucinating_generate(question: str, chunks: list) -> str:
        return "Sure, the answer is 42 and here are the detailed steps."

    with patch(
        "rag_harness.evaluation.abstention_eval.generate_async",
        side_effect=hallucinating_generate,
    ):
        result = await run_abstention_probe([_CASE])
    assert result.abstention_rate == 0.0


@pytest.mark.asyncio
async def test_run_abstention_probe_requires_distractors() -> None:
    with pytest.raises(ValueError):
        await run_abstention_probe([])
