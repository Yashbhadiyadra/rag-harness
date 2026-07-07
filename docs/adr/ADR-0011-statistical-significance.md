# ADR-0011 — Percentile bootstrap CIs on ablation metrics

**Status:** Accepted  
**Date:** 2026-07-07  
**Decided by:** Owner, 2026-07-07

## Context

The ablation table currently shows point estimates. A cell reads `0.83`
and gives the reader no way to distinguish "0.83 with a tight interval"
from "0.83 with a wide interval that overlaps 0.75." Reviewers of a
metrics-first project notice this immediately, and it makes the two
existing findings less defensible:

- **"HyDE dominates on retrieval quality."** Recall 0.90 vs. 0.77 on
  30 cases is a big-looking difference, but the CI on either estimate
  is roughly ±0.09 at n=30 with realistic variance. A reader has to
  believe that difference on faith.
- **"Corrective showed no consistent improvement across strategies."**
  The correctness delta for `dense` is +1.8pp; for `full` it is -10pp.
  Without CIs the reader cannot see that some of these deltas are inside
  small-sample noise and others are not.

The goal of this ADR is to make both of those findings quantitatively
honest: show a CI alongside every mean, and mark cross-configuration
deltas as "not statistically distinguishable from zero" when their CI
overlaps zero.

Constraints from the existing design:

- The eval-history file (`evals/history/runs.jsonl`) is append-only and
  committed to the repo. Schema changes must be backward-compatible.
- The metrics-page renderer (`scripts/render_metrics_page.py`) reads
  this file and renders a self-contained HTML page. Any per-cell CI
  must not require external libraries at render time.
- Runtime for computing a CI must be low enough to run at both eval
  time (once per run) and metrics-page render time (once per row per
  metric).

## Decision

**Adopt the non-parametric percentile bootstrap** at 95% (`alpha=0.05`,
1000 iterations) as the CI construction for every metric mean in the
ablation table and for cross-configuration paired differences.

New module `src/rag_harness/evaluation/confidence.py` with three
public functions:

- `bootstrap_ci(scores, ...) -> (lo, hi)`: CI for the mean of a set of
  per-case scores.
- `paired_diff_ci(on, off, ...) -> (lo, hi)`: CI for the mean paired
  difference. Requires case-aligned inputs.
- `ci_overlaps_zero(ci) -> bool`: convenience predicate for the
  "not statistically distinguishable from zero" message.

Stdlib-only implementation (`random`, `statistics`). Cost per call at
n=30, 1000 iterations is well under 10 ms.

### Storage: raw per-case scores in history JSONL

`HistoryEntry` gains an optional field `per_case_scores:
dict[str, list[float]] | None = None`. Keys are metric names
(`context_recall`, `context_precision`, `faithfulness`,
`correctness`, `answer_relevancy`); values are the per-case scores
as floats, aligned by case order.

Rationale: storing raw scores (rather than precomputed CIs) keeps
future analyses open. Wilcoxon signed-rank tests, McNemar tests on
pass/fail categories, and BCa bootstraps all need the raw data. A
history file that stores means + CIs cannot support those later.

Cost: each history row grows roughly 5× (five metrics × 30 floats each,
serialised as JSON). At present that's 150 additional floats per row,
under 3 KB per row after JSONL encoding. The full history at 100 runs
is under 300 KB. Negligible.

### Backfill: none. Label mixed rows explicitly.

Old rows in `evals/history/runs.jsonl` do not have per-case scores and
cannot recover them (the raw case-level output is not archived).

Rather than back-computing a fake CI, the metrics page labels old rows
`(no CI — pre-bootstrap run)` next to the point estimate. This makes
mixed rows obvious as data-provenance artefacts rather than looking
like a rendering bug.

### Display: square-bracket intervals; plain-language significance

Point-estimate + CI cells render as:

```
0.83 [0.77–0.89]
```

Square brackets are the standard "point estimate + interval" notation
in statistics papers. The dash between bounds is an en-dash (U+2013)
to match numeric-range typographic convention.

**Paired-difference significance** on the "Impact of corrective RAG"
panel switches from a green/red pp value to explicit plain language
when the CI overlaps zero:

- Positive, CI excludes zero:
  `dense: +1.8pp correctness [+0.3pp, +3.2pp], cost +$0.0039, latency +1.17s p50`
- CI overlaps zero:
  `dense: +1.8pp correctness — this difference is not statistically distinguishable from zero at n=30. cost +$0.0039, latency +1.17s p50`

The second form is the honesty win. It tells the reader, in prose, that
the number in front of them is a coin flip at the current sample size.

Old runs without per-case scores render as:
`dense: +1.8pp correctness — (no CI — pre-bootstrap run). cost +$0.0039, latency +1.17s p50`

## Alternatives considered

| Option | Why rejected |
|---|---|
| **Parametric t-CI** (`mean ± t * sem`) | Assumes normality of per-case scores. Eval scores are bounded in [0, 1] and often skewed near the ceilings. The bootstrap makes no such assumption. |
| **BCa bootstrap** (bias-corrected, accelerated) | Marginally better small-sample coverage than percentile bootstrap; substantially more code, an acceleration constant computed from jackknife. Not worth the complexity at this scale. |
| **Wilcoxon signed-rank** for paired differences | Great for hypothesis testing but produces a p-value, not a CI. We want both an effect size and its uncertainty in one number. Storing raw scores keeps Wilcoxon available as a future addition. |
| **Precompute CIs at eval time, store `(lo, hi)` per metric in history** | Cheaper at render time, but forecloses future analyses that need the raw data. Storage cost is small either way. |
| **`numpy.percentile` + `numpy.random.choice`** | Faster (~10×) for large `n_iter`, but pulls `numpy` into core deps (currently only in the `[rerank]` extra). Not worth it for 1000-iteration bootstraps at n=30. |
| **Backfill CIs on old rows via parametric approximation** | Would show a CI that does not match the bootstrap methodology and would encourage direct comparison across pre-/post-adoption rows. Better to label the gap explicitly. |

## Consequences

- Metrics page becomes materially more defensible. Every headline
  number carries its own uncertainty inline.
- The corrective-RAG panel's "green/red delta" ceases to be the whole
  story: some green deltas will show the "not distinguishable from
  zero" message. That is the honest finding, and it should reframe how
  ADR-0007's conclusions are cited.
- History JSONL rows grow ~5× (still <3 KB per row). No practical
  concern at the current volume.
- Pre-bootstrap rows in history remain valid and render with a
  clearly-labelled placeholder rather than a fake interval.
- Future analyses (Wilcoxon, McNemar, effect-size confidence bounds)
  can be added on top of the same stored data without another schema
  change.

## Implementation shape (informational)

- Commit 1: this ADR + `evaluation/confidence.py` + `tests/test_confidence.py`.
- Commit 2: extend `HistoryEntry` schema, record `per_case_scores` at
  the `record_run` boundary in both `runner.py` and `ablation.py`,
  update `tests/test_history.py` and any callers of `record_run`.
- Commit 3: extend `scripts/render_metrics_page.py` to render `[lo–hi]`
  intervals in the ablation table, show plain-language significance
  in the corrective-delta panel, and label old rows explicitly. Update
  `tests/test_render_metrics_page.py`.
