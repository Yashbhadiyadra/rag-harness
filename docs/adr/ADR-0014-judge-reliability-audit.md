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
