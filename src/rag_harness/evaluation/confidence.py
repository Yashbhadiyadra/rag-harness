"""Non-parametric percentile bootstrap confidence intervals.

Point estimates on 30 golden cases hide sampling noise. A metric mean of
0.83 could plausibly reflect anything from 0.75 to 0.90 depending on
which cases we happened to score. This module quantifies that
uncertainty as a 95% percentile bootstrap CI. See ADR-0011.

The implementation is deliberately stdlib-only. At n=30 with 1000
bootstrap iterations the cost per call is well under 10 ms, which is
fine for both the eval-time recording path and the metrics-page render
path.
"""

import random
from statistics import mean

# Public alpha default. 0.05 gives a 95% CI, the standard for reporting.
_DEFAULT_ALPHA = 0.05
_DEFAULT_N_ITER = 1000


def bootstrap_ci(
    scores: list[float],
    n_iter: int = _DEFAULT_N_ITER,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return a percentile bootstrap CI for the mean of ``scores``.

    Non-parametric: no distributional assumption on the underlying
    scores. Correct for skewed, bounded, or discrete distributions
    (all three describe RAG-eval scores).

    Parameters
    ----------
    scores:
        The per-case scores to compute the CI over.
    n_iter:
        Bootstrap iterations. 1000 is enough for two-decimal-place
        stability at typical sample sizes.
    alpha:
        Significance level. ``0.05`` gives a 95% CI (lo = 2.5th
        percentile, hi = 97.5th percentile of the bootstrap distribution).
    seed:
        If set, seeds the resampler for reproducibility. Production
        callers should leave this ``None``; tests set it to a fixed
        value.

    Returns
    -------
    ``(lo, hi)`` tuple. Edge cases:

    - Empty ``scores`` → ``(0.0, 0.0)``. Callers should check for
      empty upstream if they want to distinguish "no data" from
      "genuinely zero."
    - Single-element ``scores`` → ``(score, score)``. A one-case
      "sample" has zero variance under any bootstrap.
    """
    if not scores:
        return (0.0, 0.0)
    if len(scores) == 1:
        return (scores[0], scores[0])

    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    lo_idx = int(n_iter * (alpha / 2))
    # -1 because indices are zero-based; the 97.5th percentile of 1000
    # samples is index 974, not 975.
    hi_idx = int(n_iter * (1 - alpha / 2)) - 1
    return (means[lo_idx], means[hi_idx])


def paired_diff_ci(
    on_scores: list[float],
    off_scores: list[float],
    n_iter: int = _DEFAULT_N_ITER,
    alpha: float = _DEFAULT_ALPHA,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (on - off).

    Requires the two lists to be aligned case-by-case. Callers must
    ensure the same golden cases were scored under both configurations
    and passed in the same order.

    Returns ``(lo, hi)``. Zero-overlap in the returned interval is the
    "not statistically distinguishable from zero" signal.

    Edge cases match ``bootstrap_ci``: mismatched or empty inputs return
    ``(0.0, 0.0)``.
    """
    if not on_scores or not off_scores:
        return (0.0, 0.0)
    if len(on_scores) != len(off_scores):
        return (0.0, 0.0)

    diffs = [on - off for on, off in zip(on_scores, off_scores, strict=True)]
    return bootstrap_ci(diffs, n_iter=n_iter, alpha=alpha, seed=seed)


def ci_overlaps_zero(ci: tuple[float, float]) -> bool:
    """True when the CI includes zero (i.e., the effect is not
    statistically distinguishable from zero at the chosen alpha).

    Reads better at call sites than an inline ``lo <= 0 <= hi`` check.
    """
    lo, hi = ci
    return lo <= 0.0 <= hi


def summarise_scores(scores: list[float]) -> tuple[float, tuple[float, float]]:
    """Return ``(mean, ci)`` for a set of per-case scores.

    Convenience wrapper: the metrics page renderer calls this once per
    metric per row.
    """
    if not scores:
        return (0.0, (0.0, 0.0))
    return (mean(scores), bootstrap_ci(scores))
