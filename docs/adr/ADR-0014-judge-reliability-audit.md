# ADR-0014: Judge reliability audit

Date: 2026-07-13
Status: Accepted

## Context

Every LLM-judge score this harness reports rests on an unvalidated
instrument. Recent work quantifies the risk:

- Raw agreement rates overstate chance-corrected agreement (Cohen's
  kappa) by 34-41 percentage points across 21 judge models
  (arXiv:2606.19544).
- Judges with near-perfect test-retest consistency (>= 0.95)
  simultaneously carry severe biases - consistency alone certifies
  nothing (arXiv:2606.19544).
- No judge tested was uniformly reliable across benchmarks under
  perturbation stress-testing, and formatting perturbations degraded
  reliability more than semantic ones (arXiv:2603.05399).
- Cheaper judges matched or beat frontier judges on reliability,
  so judge selection should be evidence-based, not
  price-based (arXiv:2603.05399).

No mainstream eval tool ships judge validation as a feature. This
harness already owns the expensive prerequisites - a judge response
cache, a human-reviewed golden set, and bootstrap-CI machinery - so
auditing the judges is an incremental cost, not a new subsystem.

## Decision

Add a judge reliability audit as a first-class evaluation capability,
built in three parts:

1. **Format-invariance audit (implemented now).** For each golden
   case, score the reference answer against itself with the
   correctness judge (base score; a perfectly calibrated judge returns
   ~1.0) and with the answer-relevancy judge. Then apply deterministic,
   meaning-preserving format perturbations to the answer - code-fence
   wrapping, bullet-list restructuring, whitespace inflation, leading
   bold - and re-score. Because the perturbations change zero semantic
   content, any score shift is pure format sensitivity. Report
   per-perturbation mean/max absolute score shift and the
   verdict-flip rate at the reliability-gate thresholds.

2. **Judge provenance (implemented now).** Every audit report records
   the judge model id, a SHA-256 hash of each judge prompt, and the
   repo commit. Judge results are only comparable when
   (model, prompt) are held constant; the hashes make silent prompt
   drift detectable.

3. **Kappa against human labels (machinery now, data after review).**
   Cohen's kappa between judge verdicts and human pass/fail labels is
   the honest agreement number - raw percent agreement is banned from
   reports. The computation ships now with tests; it activates once
   golden-review decisions provide the human-label side.

Pointwise honesty: this harness's judges score single answers; they
never compare two answers in one prompt. Classic A/B position-bias
audits therefore do not apply. The analogous ordering risk here is
passage-order sensitivity in the faithfulness and context-precision
judges (their prompts embed a list of passages); auditing that
requires retrieval and is deferred to the same milestone as the
poisoned-corpus evaluation rather than half-shipped now.

The audit is an on-demand CLI command (`judge-audit`), not part of the
CI eval gate. It spends judge tokens (~30 cases x 5 variants x 2
metrics = ~300 calls, cached thereafter), and its output is a
calibration report for humans, not a pass/fail signal - gating CI on
it before baselines exist would be measurement theater.

## Consequences

- The ablation story extends from "which retrieval strategy is best"
  to "and here is the evidence the referee is fit to judge the match."
- Format sensitivity gets a number. If the correctness judge drops
  0.2 when an answer arrives wrapped in a code fence, that number
  bounds how much trust any gate threshold deserves - and motivates
  either prompt hardening or judge replacement, both now measurable.
- The audit report is publishable: per-judge, per-perturbation
  reliability numbers on a versioned golden set are exactly what the
  cited papers call for and what current tooling does not provide.
- Future judge-model comparisons (the reliability x cost selection
  matrix) reuse this machinery unchanged: run the audit under each
  candidate model, compare reports.

## First measurement (2026-07-13) and known limitation

First run (gpt-4o-mini, 30 cases, commit 377a3de): base mean 1.000 on
both metrics, zero shift and zero flips under all four perturbations.
Two readings, both recorded:

1. The production judge is perfectly calibrated on self-comparison and
   format-invariant on this probe - a real baseline worth publishing.
2. The probe has a ceiling effect. Reference-vs-itself is the easiest
   possible judging task; scores saturate at 1.0, far from the gate
   thresholds where formatting flips would materialize. Literature
   findings of format sensitivity concern borderline answers.

Follow-up (same milestone, before the public report): add a
degraded-answer probe - deterministically truncated references that
score in the 0.5-0.8 band - and measure perturbation shifts there,
where the gate actually operates. A judge that is format-invariant at
the ceiling may still flip verdicts at the boundary.

## Second measurement (2026-07-13): the boundary probe confirms it

Same judge, same 30 cases, references truncated to half their
sentences (commit 2d757ad). Base mean 0.740 - the probe landed where
intended, near the 0.80 gate. Results:

| Perturbation | Mean shift | Max shift | Verdict flips |
|---|---|---|---|
| code_fence | 0.042 | 0.250 | 10% |
| bullets | 0.037 | 0.250 | 10% |
| whitespace | 0.067 | 0.300 | 10% |
| bold_lead | 0.035 | 0.250 | 13% |

The same judge that was perfectly format-invariant at the ceiling
flips 10-13% of gate verdicts at the boundary on formatting alone,
with individual scores moving up to 0.30. This reproduces the
literature's core claim (arXiv:2603.05399) on our own golden set and
quantifies the trust radius of any single-score gate decision:
near-threshold verdicts carry roughly a ±0.07 formatting noise floor
and a one-in-ten flip risk. Consequences for the eval layer: gate
comparisons near the threshold should not be read as significant
without the bootstrap CIs (ADR-0011), and the public judge report has
its headline number.
