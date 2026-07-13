"""Tests for the judge reliability audit's pure parts (ADR-0014).

No LLM calls: perturbations, shift statistics, kappa, provenance, and
report rendering are all deterministic. The runner's LLM path is covered
by patching the per-case scorer.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from rag_harness.evaluation.judge_audit import (
    PERTURBATIONS,
    AuditReport,
    ShiftStats,
    cohens_kappa,
    prompt_hash,
    render_markdown,
    run_audit,
    score_shift_stats,
    write_report,
)
from rag_harness.models import GoldenCase

_ANSWER = "Pods are the smallest deployable unit. They wrap one or more containers."


# --- Perturbations -------------------------------------------------------


def test_perturbations_preserve_content_words() -> None:
    """Every perturbation keeps the semantic payload - only form changes."""
    for name, perturb in PERTURBATIONS.items():
        result = perturb(_ANSWER)
        for word in ("Pods", "smallest", "containers"):
            assert word in result, f"{name} lost content word {word!r}"
        assert result != _ANSWER, f"{name} was a no-op"


def test_perturbations_are_deterministic() -> None:
    for perturb in PERTURBATIONS.values():
        assert perturb(_ANSWER) == perturb(_ANSWER)


def test_code_fence_wraps() -> None:
    assert PERTURBATIONS["code_fence"](_ANSWER).startswith("```")


def test_bullets_one_per_sentence() -> None:
    result = PERTURBATIONS["bullets"](_ANSWER)
    assert result.count("- ") == 2


def test_bold_lead_single_sentence_answer() -> None:
    assert PERTURBATIONS["bold_lead"]("One sentence only") == "**One sentence only**"


# --- Shift statistics ----------------------------------------------------


def test_score_shift_stats_basic() -> None:
    stats = score_shift_stats([1.0, 0.9, 0.8], [0.9, 0.9, 0.4], threshold=0.7, perturbation="x")
    assert stats.n == 3
    assert stats.mean_abs_shift == pytest.approx((0.1 + 0.0 + 0.4) / 3)
    assert stats.max_abs_shift == pytest.approx(0.4)
    # only the third case crosses 0.7: 0.8 -> 0.4
    assert stats.flip_rate == pytest.approx(1 / 3)


def test_score_shift_stats_no_shift() -> None:
    stats = score_shift_stats([0.9, 0.9], [0.9, 0.9], threshold=0.8, perturbation="x")
    assert stats.mean_abs_shift == 0.0
    assert stats.flip_rate == 0.0


def test_score_shift_stats_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        score_shift_stats([1.0], [1.0, 0.5], threshold=0.5, perturbation="x")
    with pytest.raises(ValueError):
        score_shift_stats([], [], threshold=0.5, perturbation="x")


# --- Cohen's kappa -------------------------------------------------------


def test_kappa_perfect_agreement() -> None:
    assert cohens_kappa([True, False, True], [True, False, True]) == pytest.approx(1.0)


def test_kappa_chance_level_is_zero() -> None:
    # Independent raters, balanced labels: observed 0.5, expected 0.5.
    a = [True, True, False, False]
    b = [True, False, True, False]
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_kappa_below_chance_is_negative() -> None:
    assert cohens_kappa([True, False], [False, True]) < 0.0


def test_kappa_constant_raters() -> None:
    assert cohens_kappa([True, True], [True, True]) == 1.0
    assert cohens_kappa([True, True], [False, False]) == 0.0


def test_kappa_exposes_inflated_raw_agreement() -> None:
    """The motivating case: 90% raw agreement on skewed labels is far less
    impressive after chance correction - kappa must come out well below 0.9."""
    a = [True] * 9 + [False]
    b = [True] * 8 + [False, True]
    raw_agreement = sum(x == y for x, y in zip(a, b, strict=True)) / 10
    assert raw_agreement == pytest.approx(0.8)
    assert cohens_kappa(a, b) < 0.3


def test_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([True], [True, False])
    with pytest.raises(ValueError):
        cohens_kappa([], [])


# --- Provenance and rendering ---------------------------------------------


def test_prompt_hash_stable_and_distinct() -> None:
    assert prompt_hash("abc") == prompt_hash("abc")
    assert prompt_hash("abc") != prompt_hash("abd")
    assert len(prompt_hash("abc")) == 12


def _make_report() -> AuditReport:
    return AuditReport(
        judge_model="gpt-4o-mini",
        commit="abc1234",
        timestamp="20260713T000000+0000",
        n_cases=2,
        prompt_hashes={"correctness": "deadbeef0123"},
        base_mean={"correctness": 0.95},
        shifts={
            "correctness": [
                ShiftStats(
                    perturbation="code_fence",
                    n=2,
                    mean_abs_shift=0.10,
                    max_abs_shift=0.20,
                    flip_rate=0.5,
                )
            ]
        },
    )


def test_render_markdown_contains_provenance_and_stats() -> None:
    md = render_markdown(_make_report())
    assert "gpt-4o-mini" in md
    assert "abc1234" in md
    assert "deadbeef0123" in md
    assert "code_fence" in md
    assert "0.950" in md
    assert "50%" in md


def test_write_report_creates_md_and_json(tmp_path: Path) -> None:
    md_path = write_report(_make_report(), out_dir=tmp_path)
    assert md_path.exists()
    json_path = md_path.with_suffix(".json")
    assert json_path.exists()
    assert "judge-audit_" in md_path.name


# --- Runner (scorer patched - no LLM) --------------------------------------


@pytest.mark.asyncio
async def test_run_audit_scores_base_and_all_perturbations() -> None:
    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer=_ANSWER, relevant_doc_ids=[]),
        GoldenCase(id="t-2", question="q2", reference_answer=_ANSWER, relevant_doc_ids=[]),
    ]
    calls: list[str] = []

    async def fake_score(metric: str, case: GoldenCase, answer: str) -> float:
        calls.append(answer)
        return 1.0 if answer == case.reference_answer else 0.8

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=fake_score):
        report = await run_audit(cases=cases, metrics=("correctness",))

    # 2 cases x (1 base + 4 perturbations) = 10 scoring calls
    assert len(calls) == 10
    assert report.n_cases == 2
    assert report.base_mean["correctness"] == pytest.approx(1.0)
    stats = report.shifts["correctness"]
    assert len(stats) == len(PERTURBATIONS)
    for s in stats:
        assert s.mean_abs_shift == pytest.approx(0.2)
        assert s.flip_rate == 0.0  # 0.8 sits exactly on the 0.80 gate - still a pass
