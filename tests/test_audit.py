"""Tests for the Reliability Audit orchestrator's pure parts and wiring.

The audit adds no measurement of its own - it folds the existing probes into
one report - so these tests mock each probe's entry point and assert the
orchestrator maps their numbers to the right verdicts, grade, and report.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from rag_harness.evaluation.audit import (
    AT_RISK,
    STRONG,
    WATCH,
    Dimension,
    ReliabilityAuditReport,
    _grade,
    _verdict,
    render_markdown,
    run_reliability_audit,
)


def test_verdict_thresholds() -> None:
    assert _verdict(0.95, 0.90, 0.80) == STRONG
    assert _verdict(0.85, 0.90, 0.80) == WATCH
    assert _verdict(0.50, 0.90, 0.80) == AT_RISK
    # boundary is inclusive
    assert _verdict(0.90, 0.90, 0.80) == STRONG
    assert _verdict(0.80, 0.90, 0.80) == WATCH


def test_grade_bands() -> None:
    assert _grade(1.0).startswith("A")
    assert _grade(0.80).startswith("B")
    assert _grade(0.60).startswith("C")
    assert _grade(0.10).startswith("D")


def test_score_pct_rolls_up_verdict_points() -> None:
    report = ReliabilityAuditReport(
        commit="abc",
        timestamp="t",
        generation_model="m",
        judge_model="m",
        strategy="dense",
        n_cases=3,
        dimensions=[
            Dimension("a", "100%", STRONG, ""),  # 2
            Dimension("b", "50%", AT_RISK, ""),  # 0
        ],
    )
    # (2 + 0) / (2 * 2) = 0.5
    assert report.score_pct == pytest.approx(0.5)


def test_score_pct_empty_is_zero() -> None:
    report = ReliabilityAuditReport(
        commit="c", timestamp="t", generation_model="m", judge_model="m", strategy="d", n_cases=0
    )
    assert report.score_pct == 0.0


def test_render_markdown_includes_every_dimension_grade_and_headline() -> None:
    report = ReliabilityAuditReport(
        commit="abc123",
        timestamp="20260807T000000+0000",
        generation_model="gpt-x",
        judge_model="gpt-judge",
        strategy="hybrid",
        n_cases=10,
        dimensions=[
            Dimension("Groundedness", "94%", STRONG, "supported claims"),
            Dimension("Citation accuracy", "60%", AT_RISK, "cited passages"),
        ],
        grade="B - Largely trustworthy",
        headline="Your own evaluator overstates agreement by 6 points.",
        quality_pass_cost_usd=0.1234,
    )
    md = render_markdown(report)
    assert "# Reliability Audit" in md
    assert "abc123" in md
    assert "B - Largely trustworthy" in md
    assert "Your own evaluator overstates agreement by 6 points." in md
    assert "Groundedness" in md and "94%" in md
    assert "Citation accuracy" in md and "60%" in md
    assert "$0.1234" in md


def _fake_summary() -> SimpleNamespace:
    return SimpleNamespace(
        mean_faithfulness=0.94,
        mean_correctness=0.88,
        total_cost_usd=0.05,
    )


def _patches(*, with_kappa: bool = True):
    """Patch every probe entry point the orchestrator calls."""
    kappa = SimpleNamespace(raw_agreement=0.94, kappa=0.88) if with_kappa else None
    audit_report = SimpleNamespace(
        judge_model="gpt-judge",
        kappa=kappa,
        scales=[SimpleNamespace(flip_rate=0.10)],
    )
    cases = [SimpleNamespace(id="c1"), SimpleNamespace(id="c2")]
    return {
        "load_golden_cases": patch(
            "rag_harness.evaluation.audit.load_golden_cases", return_value=cases
        ),
        "git": patch("rag_harness.evaluation.audit._current_git_commit", return_value="deadbee"),
        "build_retriever": patch(
            "rag_harness.retrieval.factory.build_retriever", return_value=object()
        ),
        "run_eval": patch(
            "rag_harness.evaluation.runner.run_eval",
            new=AsyncMock(return_value=_fake_summary()),
        ),
        "citation": patch(
            "rag_harness.evaluation.citation_eval.run_citation_eval_sync",
            return_value=SimpleNamespace(accuracy=0.90),
        ),
        "abstention": patch(
            "rag_harness.evaluation.abstention_eval.run_abstention_probe_sync",
            return_value=SimpleNamespace(abstention_rate=1.0),
        ),
        "security": patch(
            "rag_harness.evaluation.security_eval.run_security_eval_sync",
            return_value=(
                [SimpleNamespace(resistance_rate=1.0)],
                [SimpleNamespace(resistance_rate=0.95)],
            ),
        ),
        "judge": patch(
            "rag_harness.evaluation.judge_audit.run_audit_sync",
            return_value=audit_report,
        ),
    }


def test_run_reliability_audit_full_builds_all_dimensions() -> None:
    p = _patches(with_kappa=True)
    with (
        p["load_golden_cases"],
        p["git"],
        p["build_retriever"],
        p["run_eval"],
        p["citation"],
        p["abstention"],
        p["security"],
        p["judge"],
    ):
        report = run_reliability_audit(strategy="dense", sample=2, include_judge=True)

    names = [d.name for d in report.dimensions]
    # 6 base dimensions + 2 judge dimensions (kappa + format-flip)
    assert len(report.dimensions) == 8
    assert any("Judge trustworthiness" in n for n in names)
    assert any("format-flip" in n for n in names)
    assert report.headline  # signature callout is set when kappa is present
    assert "overstates agreement by 6 points" in report.headline
    assert report.judge_model == "gpt-judge"
    assert report.commit == "deadbee"
    assert report.grade  # a grade is assigned


def test_run_reliability_audit_no_judge_skips_judge_dimensions() -> None:
    p = _patches()
    with (
        p["load_golden_cases"],
        p["git"],
        p["build_retriever"],
        p["run_eval"],
        p["citation"],
        p["abstention"],
        p["security"],
    ):
        report = run_reliability_audit(strategy="dense", sample=2, include_judge=False)

    assert len(report.dimensions) == 6
    assert report.headline == ""
    assert all("Judge" not in d.name for d in report.dimensions)


def test_run_reliability_audit_separates_injection_from_data_poisoning() -> None:
    """Injection and counterfactual are distinct: conflating them would mislabel
    the by-design data-poisoning stance as an injection vulnerability."""
    p = _patches(with_kappa=False)
    with (
        p["load_golden_cases"],
        p["git"],
        p["build_retriever"],
        p["run_eval"],
        p["citation"],
        p["abstention"],
        p["security"],  # injection resistance 1.0, counterfactual 0.95
    ):
        report = run_reliability_audit(sample=2, include_judge=False)

    injection = next(d for d in report.dimensions if "Injection" in d.name)
    counterfactual = next(d for d in report.dimensions if "Counterfactual" in d.name)
    assert injection.value == "100%"
    assert counterfactual.value == "95%"
