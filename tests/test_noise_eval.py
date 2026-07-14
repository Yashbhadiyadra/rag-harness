"""Tests for the noise-robustness probe's pure parts."""

from unittest.mock import patch

import pytest

from rag_harness.evaluation.noise_eval import (
    NoiseLevelResult,
    build_noisy_context,
    degradation,
    render_markdown,
    run_noise_probe,
)
from rag_harness.models import GoldenCase


def _case(i: int) -> GoldenCase:
    return GoldenCase(
        id=f"c-{i}",
        question=f"question {i}?",
        reference_answer=f"reference answer {i}",
        relevant_doc_ids=[f"doc-{i}.md"],
    )


def test_build_noisy_context_correct_chunk_first() -> None:
    cases = [_case(0), _case(1), _case(2)]
    ctx = build_noisy_context(cases[0], cases[1:], k=2)
    assert len(ctx) == 3  # 1 correct + 2 distractors
    assert "reference answer 0" in ctx[0].text
    # distractors carry other cases' content
    assert "reference answer 1" in ctx[1].text


def test_build_noisy_context_k_zero_is_correct_only() -> None:
    ctx = build_noisy_context(_case(0), [_case(1)], k=0)
    assert len(ctx) == 1
    assert "reference answer 0" in ctx[0].text


def test_build_noisy_context_cycles_when_short() -> None:
    ctx = build_noisy_context(_case(0), [_case(1)], k=3)
    # only one distractor available -> cycled, so 1 correct + 3 copies
    assert len(ctx) == 4


def test_degradation_computes_drop() -> None:
    results = [
        NoiseLevelResult(k=0, n_cases=30, mean_faithfulness=0.90, mean_correctness=0.80),
        NoiseLevelResult(k=5, n_cases=30, mean_faithfulness=0.82, mean_correctness=0.70),
    ]
    faith_drop, correct_drop = degradation(results)
    assert faith_drop == pytest.approx(0.08)
    assert correct_drop == pytest.approx(0.10)


def test_degradation_single_level_is_zero() -> None:
    results = [NoiseLevelResult(k=0, n_cases=30, mean_faithfulness=0.9, mean_correctness=0.8)]
    assert degradation(results) == (0.0, 0.0)


def test_render_markdown_has_row_per_level() -> None:
    results = [
        NoiseLevelResult(k=0, n_cases=30, mean_faithfulness=0.90, mean_correctness=0.80),
        NoiseLevelResult(k=2, n_cases=30, mean_faithfulness=0.88, mean_correctness=0.78),
        NoiseLevelResult(k=5, n_cases=30, mean_faithfulness=0.82, mean_correctness=0.70),
    ]
    md = render_markdown(results, commit="abc1234", timestamp="20260714T000000+0000")
    assert "Noise robustness" in md
    assert "abc1234" in md
    assert "0.900" in md and "0.820" in md
    assert "Degradation" in md


@pytest.mark.asyncio
async def test_run_noise_probe_produces_a_result_per_level() -> None:
    cases = [_case(0), _case(1), _case(2)]

    async def fake_generate(question: str, chunks: list) -> str:
        return "some answer"

    async def fake_faith(question: str, answer: str, chunks: list) -> float:
        # faithfulness falls as more chunks are present
        return max(0.0, 1.0 - 0.05 * len(chunks))

    async def fake_correct(question: str, answer: str, reference: str) -> float:
        return 0.8

    with (
        patch("rag_harness.evaluation.noise_eval.generate_async", side_effect=fake_generate),
        patch("rag_harness.evaluation.noise_eval.faithfulness_async", side_effect=fake_faith),
        patch("rag_harness.evaluation.noise_eval.correctness_async", side_effect=fake_correct),
    ):
        results = await run_noise_probe(cases, noise_levels=(0, 2))

    assert [r.k for r in results] == [0, 2]
    # k=0 has 1 chunk (faith 0.95), k=2 has 3 chunks (faith 0.85) -> degradation
    assert results[0].mean_faithfulness > results[1].mean_faithfulness
    assert results[0].mean_correctness == pytest.approx(0.8)
