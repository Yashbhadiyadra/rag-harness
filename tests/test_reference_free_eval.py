"""Tests for the reference-free detection probe's pure parts."""

from unittest.mock import patch

import pytest

from rag_harness.evaluation.reference_free_eval import (
    ReferenceFreeResult,
    detection_stats,
    render_markdown,
    run_reference_free_probe,
)
from rag_harness.models import GoldenCase


def _case(i: int) -> GoldenCase:
    return GoldenCase(
        id=f"c-{i}",
        question=f"question {i}?",
        reference_answer=f"reference answer {i}",
        relevant_doc_ids=[f"doc-{i}.md"],
    )


def test_detection_stats_perfect_separation() -> None:
    grounded = [1.0, 0.9, 1.0]
    ungrounded = [0.1, 0.0, 0.2]
    r = detection_stats(grounded, ungrounded)
    assert r.mean_faithful_grounded == pytest.approx(0.9667, abs=1e-3)
    assert r.mean_faithful_ungrounded == pytest.approx(0.1, abs=1e-3)
    assert r.separation == pytest.approx(0.8667, abs=1e-3)
    assert r.detection_accuracy == 1.0  # all 6 classified correctly


def test_detection_stats_no_separation() -> None:
    # both populations straddle the threshold identically -> chance accuracy
    r = detection_stats([0.6, 0.4], [0.6, 0.4])
    assert r.separation == 0.0
    # grounded: 0.6>=.5 ok, 0.4<.5 wrong; ungrounded: 0.6<.5? no ->wrong, 0.4<.5 ok
    assert r.detection_accuracy == 0.5


def test_detection_stats_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        detection_stats([0.9], [0.1, 0.2])
    with pytest.raises(ValueError):
        detection_stats([], [])


def test_render_markdown_reports_separation_and_hhem_note() -> None:
    r = ReferenceFreeResult(
        n_cases=30,
        mean_faithful_grounded=0.95,
        mean_faithful_ungrounded=0.20,
        detection_accuracy=0.93,
    )
    md = render_markdown(r, commit="abc1234", timestamp="20260714T000000+0000")
    assert "Reference-free" in md
    assert "abc1234" in md
    assert "0.750" in md  # separation
    assert "93%" in md
    assert "HHEM" in md  # future-work note present


@pytest.mark.asyncio
async def test_run_reference_free_probe_separates_populations() -> None:
    cases = [_case(0), _case(1), _case(2)]

    async def fake_faith(question: str, answer: str, chunks: list) -> float:
        # grounded when the answer text matches the context chunk's content
        ctx = chunks[0].text
        return 1.0 if answer == ctx else 0.1

    with patch(
        "rag_harness.evaluation.reference_free_eval.faithfulness_async", side_effect=fake_faith
    ):
        result = await run_reference_free_probe(cases)

    assert result.mean_faithful_grounded == pytest.approx(1.0)
    assert result.mean_faithful_ungrounded == pytest.approx(0.1)
    assert result.detection_accuracy == 1.0


@pytest.mark.asyncio
async def test_run_reference_free_probe_needs_two_cases() -> None:
    with pytest.raises(ValueError):
        await run_reference_free_probe([_case(0)])
