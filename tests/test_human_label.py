"""Tests for the human-label judge-validation harness (ADR-0014 follow-up)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rag_harness.evaluation.human_label import (
    LabelItem,
    build_label_sample,
    load_sample,
    score_human_kappa,
    write_sample,
)
from rag_harness.models import GoldenCase


def _cases() -> list[GoldenCase]:
    return [
        GoldenCase(id="c1", question="q1?", reference_answer="ref1", relevant_doc_ids=[]),
        GoldenCase(id="c2", question="q2?", reference_answer="ref2", relevant_doc_ids=[]),
        GoldenCase(id="c3", question="q3?", reference_answer="ref3", relevant_doc_ids=[]),
    ]


def test_build_label_sample_is_balanced_and_deterministic() -> None:
    with patch("rag_harness.evaluation.human_label.load_golden_cases", return_value=_cases()):
        items = build_label_sample(2)
        items2 = build_label_sample(2)

    assert len(items) == 4  # 2 cases * (correct + crosspair)
    assert [i.id for i in items] == [i.id for i in items2]  # deterministic
    correct = items[0]
    crosspair = items[1]
    assert correct.expected == "correct"
    assert correct.answer == correct.reference == "ref1"
    assert crosspair.expected == "incorrect"
    assert crosspair.answer == "ref2"  # next case's reference
    assert crosspair.reference == "ref1"  # judged against this question's reference


def test_write_and_load_sample_roundtrip(tmp_path: Path) -> None:
    with patch("rag_harness.evaluation.human_label.load_golden_cases", return_value=_cases()):
        items = build_label_sample(2)
    path = tmp_path / "sample.jsonl"
    write_sample(items, path)
    loaded = load_sample(path)
    assert loaded == items
    assert all(i.human_label == "" for i in loaded)  # blank, awaiting the human


@pytest.mark.asyncio
async def test_score_human_kappa_perfect_agreement() -> None:
    items = [
        LabelItem("a", "q?", "ref", "ref", "correct", human_label="correct"),
        LabelItem("b", "q?", "ref", "wrong", "incorrect", human_label="incorrect"),
    ]

    async def fake_correctness(question: str, answer: str, reference: str) -> float:
        return 0.95 if answer == "ref" else 0.1

    with patch("rag_harness.evaluation.metrics.correctness_async", side_effect=fake_correctness):
        result = await score_human_kappa(items)

    assert result.n_labeled == 2
    assert result.judge_human_disagreements == 0
    assert result.raw_agreement == 1.0
    assert result.kappa == 1.0


@pytest.mark.asyncio
async def test_score_human_kappa_counts_disagreement() -> None:
    # Human says the second answer is correct, but the judge scores it low.
    items = [
        LabelItem("a", "q?", "ref", "ref", "correct", human_label="correct"),
        LabelItem("b", "q?", "ref", "wrong", "incorrect", human_label="correct"),
    ]

    async def fake_correctness(question: str, answer: str, reference: str) -> float:
        return 0.95 if answer == "ref" else 0.1

    with patch("rag_harness.evaluation.metrics.correctness_async", side_effect=fake_correctness):
        result = await score_human_kappa(items)

    assert result.judge_human_disagreements == 1
    assert result.raw_agreement == 0.5


@pytest.mark.asyncio
async def test_score_human_kappa_skips_unlabeled_and_errors_when_none() -> None:
    items = [LabelItem("a", "q?", "ref", "ref", "correct", human_label="")]
    with pytest.raises(ValueError, match="no items"):
        await score_human_kappa(items)
