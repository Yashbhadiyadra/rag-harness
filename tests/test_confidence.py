"""Tests for the bootstrap-CI module.

Deterministic via ``seed=`` parameter. No OpenAI, no network. See ADR-0011.
"""

import pytest

from rag_harness.evaluation.confidence import (
    bootstrap_ci,
    ci_overlaps_zero,
    paired_diff_ci,
    summarise_scores,
)

# ---- Edge cases ------------------------------------------------------


def test_empty_scores_returns_zero_ci() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_single_score_returns_degenerate_ci() -> None:
    """One-case sample has zero variance under any bootstrap."""
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


# ---- Basic correctness -----------------------------------------------


def test_bootstrap_ci_contains_true_mean_at_typical_n() -> None:
    """CI should contain the population mean for a well-behaved sample.

    Scores drawn tightly around 0.8: the CI should be a narrow band
    that includes 0.8. This is a smoke test of the whole pipeline,
    not a coverage test.
    """
    scores = [0.75, 0.78, 0.80, 0.82, 0.85] * 6  # n=30
    lo, hi = bootstrap_ci(scores, seed=42)
    assert lo <= 0.80 <= hi
    # Sanity: bounds are reasonable (not degenerate, not absurdly wide)
    assert 0.76 <= lo < 0.80
    assert 0.80 < hi <= 0.84


def test_wider_variance_produces_wider_ci() -> None:
    """A high-variance sample must give a wider CI than a low-variance one
    at the same n. This is the core reason the CI exists."""
    tight = [0.79, 0.80, 0.81] * 10  # very low variance around 0.8
    wide = [0.30, 0.80, 0.99] * 10  # bimodal around the same mean
    tight_lo, tight_hi = bootstrap_ci(tight, seed=1)
    wide_lo, wide_hi = bootstrap_ci(wide, seed=1)
    assert (wide_hi - wide_lo) > (tight_hi - tight_lo)


# ---- Determinism -----------------------------------------------------


def test_same_seed_produces_same_ci() -> None:
    scores = [0.6, 0.7, 0.8, 0.9] * 8
    ci1 = bootstrap_ci(scores, seed=123)
    ci2 = bootstrap_ci(scores, seed=123)
    assert ci1 == ci2


def test_different_seed_can_produce_different_ci() -> None:
    """Sanity: seed matters. On a continuous-ish sample two very
    different seeds should almost never produce byte-identical CIs."""
    scores = [0.13, 0.27, 0.31, 0.42, 0.56, 0.68, 0.71, 0.83, 0.92, 0.95]
    ci1 = bootstrap_ci(scores, seed=1, n_iter=200)
    ci2 = bootstrap_ci(scores, seed=99999, n_iter=200)
    assert ci1 != ci2


# ---- Paired difference ------------------------------------------------


def test_paired_diff_zero_effect_ci_includes_zero() -> None:
    """When on_scores == off_scores, every paired diff is exactly zero
    and the CI must include zero."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
    lo, hi = paired_diff_ci(scores, scores, seed=42)
    assert lo == 0.0 and hi == 0.0
    assert ci_overlaps_zero((lo, hi))


def test_paired_diff_uniform_positive_effect_ci_excludes_zero() -> None:
    """When on_scores are strictly higher by a large fixed amount, the
    paired diff CI must be strictly positive at n=30."""
    off = [0.5, 0.55, 0.6, 0.65, 0.7] * 6
    on = [x + 0.15 for x in off]
    lo, hi = paired_diff_ci(on, off, seed=42)
    assert lo > 0.0
    assert hi > 0.0
    assert not ci_overlaps_zero((lo, hi))


def test_paired_diff_mismatched_lengths_returns_zero_ci() -> None:
    assert paired_diff_ci([1.0, 2.0], [1.0]) == (0.0, 0.0)


def test_paired_diff_empty_returns_zero_ci() -> None:
    assert paired_diff_ci([], []) == (0.0, 0.0)


# ---- summarise_scores wrapper ----------------------------------------


def test_summarise_scores_returns_mean_and_ci() -> None:
    scores = [0.7, 0.8, 0.9] * 10
    m, (lo, hi) = summarise_scores(scores)
    assert m == pytest.approx(0.8)
    assert lo <= 0.8 <= hi


def test_summarise_scores_empty() -> None:
    assert summarise_scores([]) == (0.0, (0.0, 0.0))
