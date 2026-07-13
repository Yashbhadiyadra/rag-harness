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
    ProbeResult,
    ShiftStats,
    cohens_kappa,
    cross_pair_answers,
    prompt_hash,
    render_markdown,
    run_audit,
    score_shift_stats,
    stability_stats,
    truncate_half,
    verbose_pad,
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


# --- Degradation ---------------------------------------------------------


def test_truncate_half_keeps_first_half() -> None:
    text = "First point. Second point. Third point. Fourth point."
    result = truncate_half(text)
    assert result == "First point. Second point."


def test_truncate_half_single_sentence_keeps_it() -> None:
    assert truncate_half("Only sentence here.") == "Only sentence here."


def test_truncate_half_is_deterministic_and_strictly_shorter() -> None:
    assert truncate_half(_ANSWER) == truncate_half(_ANSWER)
    assert len(truncate_half(_ANSWER)) < len(_ANSWER)


def test_cross_pair_answers_rotates_references() -> None:
    cases = [
        GoldenCase(id=f"t-{i}", question=f"q{i}", reference_answer=f"ref{i}", relevant_doc_ids=[])
        for i in range(3)
    ]
    assert cross_pair_answers(cases) == ["ref1", "ref2", "ref0"]


def test_cross_pair_answers_needs_two_cases() -> None:
    only = [GoldenCase(id="t", question="q", reference_answer="r", relevant_doc_ids=[])]
    with pytest.raises(ValueError):
        cross_pair_answers(only)


def test_verbose_pad_extends_without_removing_content() -> None:
    padded = verbose_pad(_ANSWER)
    assert padded.startswith(_ANSWER.rstrip()[:20])
    assert len(padded) > len(_ANSWER)
    for word in ("Pods", "containers"):
        assert word in padded


def test_verbose_pad_is_deterministic() -> None:
    assert verbose_pad(_ANSWER) == verbose_pad(_ANSWER)


# --- Shift statistics ----------------------------------------------------


def test_score_shift_stats_basic() -> None:
    stats = score_shift_stats([1.0, 0.9, 0.8], [0.9, 0.9, 0.4], threshold=0.7, perturbation="x")
    assert stats.n == 3
    assert stats.mean_abs_shift == pytest.approx((0.1 + 0.0 + 0.4) / 3)
    assert stats.max_abs_shift == pytest.approx(0.4)
    # only the third case crosses 0.7: 0.8 -> 0.4
    assert stats.flip_rate == pytest.approx(1 / 3)
    # signed shift keeps direction: (0.9-1.0)+(0.9-0.9)+(0.4-0.8) = -0.5, /3
    assert stats.signed_mean_shift == pytest.approx(-0.5 / 3)


def test_score_shift_stats_signed_positive_is_inflation() -> None:
    # every perturbed score is higher: pure inflation (verbosity-bias shape)
    stats = score_shift_stats([0.6, 0.7], [0.8, 0.9], threshold=0.75, perturbation="verbose_pad")
    assert stats.signed_mean_shift == pytest.approx(0.2)
    assert stats.mean_abs_shift == pytest.approx(0.2)


def test_score_shift_stats_no_shift() -> None:
    stats = score_shift_stats([0.9, 0.9], [0.9, 0.9], threshold=0.8, perturbation="x")
    assert stats.mean_abs_shift == 0.0
    assert stats.flip_rate == 0.0


def test_score_shift_stats_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        score_shift_stats([1.0], [1.0, 0.5], threshold=0.5, perturbation="x")
    with pytest.raises(ValueError):
        score_shift_stats([], [], threshold=0.5, perturbation="x")


# --- Test-retest stability -----------------------------------------------


def test_stability_stats_perfectly_stable() -> None:
    # every case scored identically across repeats -> zero variance
    stats = stability_stats([[0.7, 0.7, 0.7], [0.9, 0.9, 0.9]], "correctness", 3)
    assert stats.mean_variance == 0.0
    assert stats.max_within_case_range == 0.0
    assert stats.n_unstable_cases == 0
    assert stats.n_cases == 2


def test_stability_stats_flags_inconsistent_case() -> None:
    stats = stability_stats([[0.7, 0.7], [0.6, 0.9]], "correctness", 2)
    assert stats.max_within_case_range == pytest.approx(0.3)
    assert stats.n_unstable_cases == 1
    assert stats.mean_variance > 0.0


def test_stability_stats_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        stability_stats([[0.7]], "correctness", 1)  # repeats < 2
    with pytest.raises(ValueError):
        stability_stats([[0.7, 0.7], [0.8]], "correctness", 2)  # ragged
    with pytest.raises(ValueError):
        stability_stats([], "correctness", 2)  # empty


@pytest.mark.asyncio
async def test_run_audit_retest_adds_stability_and_disables_cache() -> None:
    from rag_harness.config import settings

    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer=_ANSWER, relevant_doc_ids=[]),
    ]
    settings.llm_cache_enabled = True  # must be restored by run_stability
    seen_cache_flags: list[bool] = []

    async def fake_score(metric: str, case: GoldenCase, answer: str) -> float:
        seen_cache_flags.append(settings.llm_cache_enabled)
        return 0.7

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=fake_score):
        report = await run_audit(cases=cases, probes=("boundary",), retest=3)

    assert len(report.stability) == 1
    st = report.stability[0]
    assert st.repeats == 3 and st.n_cases == 1
    assert st.mean_variance == 0.0  # constant fake score
    # the retest scoring must have run with the cache OFF...
    assert any(flag is False for flag in seen_cache_flags)
    # ...and the original setting restored afterward
    assert settings.llm_cache_enabled is True


@pytest.mark.asyncio
async def test_run_audit_no_retest_by_default() -> None:
    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer=_ANSWER, relevant_doc_ids=[]),
    ]

    async def fake_score(metric: str, case: GoldenCase, answer: str) -> float:
        return 0.7

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=fake_score):
        report = await run_audit(cases=cases, probes=("boundary",))
    assert report.stability == []


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
        probes=[
            ProbeResult(
                probe="boundary",
                metric="correctness",
                base_mean=0.95,
                base_pass_rate=0.5,
                shifts=[
                    ShiftStats(
                        perturbation="code_fence",
                        n=2,
                        mean_abs_shift=0.10,
                        max_abs_shift=0.20,
                        flip_rate=0.5,
                    )
                ],
            )
        ],
    )


def test_render_markdown_contains_provenance_and_stats() -> None:
    md = render_markdown(_make_report())
    assert "gpt-4o-mini" in md
    assert "abc1234" in md
    assert "deadbeef0123" in md
    assert "code_fence" in md
    assert "boundary · correctness" in md
    assert "0.950" in md
    assert "pass rate at gate: **50%**" in md


def test_write_report_creates_md_and_json(tmp_path: Path) -> None:
    md_path = write_report(_make_report(), out_dir=tmp_path)
    assert md_path.exists()
    json_path = md_path.with_suffix(".json")
    assert json_path.exists()
    assert "judge-audit_" in md_path.name


# --- Runner (scorer patched - no LLM) --------------------------------------


@pytest.mark.asyncio
async def test_run_audit_ceiling_probe_scores_base_and_all_perturbations() -> None:
    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer=_ANSWER, relevant_doc_ids=[]),
        GoldenCase(id="t-2", question="q2", reference_answer=_ANSWER, relevant_doc_ids=[]),
    ]
    calls: list[str] = []

    async def fake_score(metric: str, case: GoldenCase, answer: str) -> float:
        calls.append(answer)
        return 1.0 if answer == case.reference_answer else 0.8

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=fake_score):
        report = await run_audit(cases=cases, probes=("ceiling",))

    # ceiling covers 2 metrics: 2 cases x (1 base + 4 perturbations) x 2 = 20 calls
    assert len(calls) == 20
    assert report.n_cases == 2
    correctness = next(
        r for r in report.probes if r.probe == "ceiling" and r.metric == "correctness"
    )
    assert correctness.base_mean == pytest.approx(1.0)
    # ceiling gets format perturbations only - no verbose_pad (no headroom at 1.0)
    assert len(correctness.shifts) == len(PERTURBATIONS)
    assert all(s.perturbation != "verbose_pad" for s in correctness.shifts)
    for s in correctness.shifts:
        assert s.mean_abs_shift == pytest.approx(0.2)
        assert s.flip_rate == 0.0  # 0.8 sits exactly on the 0.80 gate - still a pass


@pytest.mark.asyncio
async def test_run_audit_boundary_probe_judges_truncated_answers() -> None:
    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer=_ANSWER, relevant_doc_ids=[]),
    ]
    seen_answers: list[str] = []

    async def fake_score(metric: str, case: GoldenCase, answer: str) -> float:
        seen_answers.append(answer)
        return 0.75

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=fake_score):
        report = await run_audit(cases=cases, probes=("boundary",))

    # boundary correctness: 1 case x (1 base + 4 format + 1 verbose_pad) = 6 calls
    assert len(seen_answers) == 6
    truncated = truncate_half(_ANSWER)
    assert seen_answers[0] == truncated
    # every variant derives from the truncated answer, not the full one
    assert all("They wrap" not in a for a in seen_answers)
    assert [r.metric for r in report.probes] == ["correctness"]
    # boundary carries the verbosity probe
    assert any(s.perturbation == "verbose_pad" for s in report.probes[0].shifts)
    # 0.75 sits below the 0.80 correctness gate - nothing passes
    assert report.probes[0].base_pass_rate == 0.0


@pytest.mark.asyncio
async def test_run_audit_boundary_detects_verbosity_inflation() -> None:
    """A length-biased judge scores the padded answer higher; the signed
    shift on verbose_pad must be positive and flag it."""
    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer=_ANSWER, relevant_doc_ids=[]),
    ]

    async def length_biased_score(metric: str, case: GoldenCase, answer: str) -> float:
        # longer answer -> higher score, regardless of content
        return min(1.0, 0.5 + len(answer) / 2000)

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=length_biased_score):
        report = await run_audit(cases=cases, probes=("boundary",))

    verbose = next(s for s in report.probes[0].shifts if s.perturbation == "verbose_pad")
    assert verbose.signed_mean_shift > 0.0  # padding inflated the score


@pytest.mark.asyncio
async def test_run_audit_discrimination_probe_reports_false_accept_rate() -> None:
    cases = [
        GoldenCase(id="t-1", question="q1", reference_answer="About pods.", relevant_doc_ids=[]),
        GoldenCase(id="t-2", question="q2", reference_answer="About rbac.", relevant_doc_ids=[]),
    ]

    async def fake_score(metric: str, case: GoldenCase, answer: str) -> float:
        # A discriminating judge: low score when the answer is the other
        # case's reference, regardless of formatting.
        return 0.1 if case.reference_answer.split()[1] not in answer else 1.0

    with patch("rag_harness.evaluation.judge_audit._score_case", side_effect=fake_score):
        report = await run_audit(cases=cases, probes=("discrimination",))

    result = report.probes[0]
    assert result.probe == "discrimination"
    assert result.base_mean == pytest.approx(0.1)
    assert result.base_pass_rate == 0.0  # no wrong answer passed the gate
