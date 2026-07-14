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

## Third measurement (2026-07-13): discrimination probe

Format noise alone cannot certify a judge - a judge that passes
everything also shows zero flips. The discrimination probe (JRH's
label-flip accuracy) cross-pairs each question with another case's
reference: fluent, on-domain, wrong by construction. Results
(gpt-4o-mini, 30 cases, commit b787f0a):

- Base mean 0.027, **false-accept rate 0%** - no wrong answer passed
  the 0.80 gate.
- Under all four format perturbations: 0% flips - **formatting cannot
  rescue a wrong answer** (max individual shift 0.30, but never across
  the gate).

Complete three-probe verdict for the production judge: calibrated at
the ceiling (1.000), discriminating at the floor (0% false accepts),
noisy only in the middle (10-13% flips near the gate). The middle is
where pipeline outputs live, which is why the bootstrap CIs stay
mandatory. This is the shape of the public judge report: three probes,
three numbers, one honest conclusion.

## Fourth measurement (2026-07-13): verbosity-bias probe

The classic LLM-judge failure is verbosity bias: rewarding longer
answers over substance. The probe appends deterministic content-free
filler (hedging prose that adds zero correct information) to the
boundary and discrimination answers and reports the *signed* mean
shift - positive would mean the padding inflated the score.

Result on the boundary probe (gpt-4o-mini, 30 cases): signed shift
**-0.180**, with 23% of gate verdicts flipping - almost all
*downward*. The production judge is not verbosity-biased; it does the
opposite, penalising a partial answer that gets padded with fluff
(the filler dilutes the on-point content and reads as non-answer).
On discrimination answers the effect is negligible (-0.013). To make
the direction legible, `ShiftStats` now carries `signed_mean_shift`
alongside the absolute magnitude.

This is a genuinely good property worth publishing: the gate cannot be
gamed by padding. It also sharpens the boundary caution - verbose_pad
is the single largest perturbation (0.180 mean shift vs 0.03-0.07 for
formatting), so answer length near the threshold moves scores far more
than cosmetic formatting does.

## Fifth measurement (2026-07-13): test-retest stability

`--retest 5` judges the boundary answers five times each with the
cache off; any spread is the judge disagreeing with itself. At
temperature 0 the naive expectation is zero variance. Result
(gpt-4o-mini, 30 cases): mean variance **0.0011**, but **5 of 30
cases were unstable**, one with a within-case range of **0.250**.

Temperature 0 does not guarantee reproducibility - API-level
nondeterminism gives the same judge, same prompt, same answer
different scores about one time in six near the gate. Most cases
(25/30) are rock-stable, but the unstable minority carries a range as
large as the format and length effects combined. This is a third,
independent noise source stacked on formatting (~0.07) and length
(0.18): a single judge score near the threshold should never be read
as exact. The operational consequence is unchanged and reinforced -
gate decisions live or die by the ADR-0011 bootstrap CIs, and the
eval layer should judge with the cache on (pinning one score per
input) so this nondeterminism does not leak into run-to-run
comparisons.

## Sixth measurement (2026-07-13): scale-format sensitivity

CIP found the response scale alone can move a judge's verdict (1.68 on
a 1-5 scale vs 3.17 on A-E for the same item). `--scales` scores the
boundary answers on the numeric [0,1] correctness scale and an
equivalent A-E letter scale (A=1.0 .. E=0.0) and compares. Result
(gpt-4o-mini, 30 cases): mean absolute divergence **0.090**, signed
**-0.057** (the letter scale grades stricter), and **23% of gate
verdicts flip** between scales.

Nearly a quarter of pass/fail decisions depend on whether the judge is
asked for a number or a letter. Part of this is the letter scale's
coarse 0.25-wide buckets versus continuous numeric scoring, but the
consistent negative sign shows a real stricter-grading bias, not just
quantization. This is the fourth and largest-flip-rate noise source
found, and it argues for pinning one scale (we keep numeric) and never
mixing scales within a comparison. It is also the strongest single
entry for the public report: the scale you pick is not a cosmetic
choice, it moves a fifth of the verdicts.

## Seventh measurement (2026-07-14): judge selection matrix

`judge-matrix` runs the full audit under each candidate model and lays
the four decision numbers side by side. Cache off, so cost is real and
comparable. Result:

| Judge model | Ceiling calib | Worst boundary flip | False-accept | Cost (USD) |
|---|---|---|---|---|
| gpt-4o-mini | 1.000 | 23% | 0% | 0.0288 |
| gpt-4o | 1.000 | 13% | 0% | 0.4799 |

Both models are perfectly calibrated (ceiling 1.0) and perfectly
discriminating (0% false-accept). They differ only in boundary
robustness: gpt-4o flips 13% of near-gate verdicts under formatting
versus gpt-4o-mini's 23% - genuinely more robust - but costs **16.7x
more** per audit.

This did NOT reproduce the generic "a cheap judge always matches an
expensive one" claim: here the expensive model is measurably better at
the boundary. That is the point of the matrix - reliability is model-
and task-dependent, so the choice must be measured, not assumed. The
evidence-based call for this project: gpt-4o-mini stays the default
(the 23% boundary noise is mitigated by cache-on scoring, which pins
one score per input, plus the ADR-0011 bootstrap CIs), and gpt-4o is
the option when near-gate precision must be maximised and cost is
secondary. The matrix also surfaced a real bug on first run - gpt-4o
was unpriced and reported a false $0.00 cost - now fixed in the
pricing table.

Note the boundary flip rates carry the test-retest nondeterminism
measured above (gpt-4o showed 7% on one run, 13% on another): near-gate
flip rates are themselves noisy, which is exactly why single runs are
never read as exact and the CIs are mandatory.

## Eighth measurement (2026-07-14): judge-vs-truth kappa

The literature's headline demand is chance-corrected agreement (Cohen's
kappa), not raw agreement (arXiv:2606.19544). `--kappa` measures it on
the expanded 160-case set against a known ground truth: each golden
reference answer is known-correct, each cross-paired answer (case i
answered with case i+1's reference) is known-incorrect. The judge's
correctness gate verdict is compared to that truth.

Result (gpt-4o-mini, 160 known-correct + 160 known-incorrect):

| Cohen's kappa | Raw agreement | False accepts | False rejects |
|---|---|---|---|
| **0.875** | 0.938 | 20/160 | 0/160 |

Three honest readings:

1. Kappa 0.875 is substantial agreement, and raw agreement (0.938)
   overstates it by 6.3 points - demonstrating the field's core warning
   on our own judge, exactly why we report kappa and not raw agreement.
2. Zero false rejects: the judge never fails a correct golden answer, so
   the gate does not punish good answers. That is the property that
   matters most for a quality gate.
3. Twelve percent false accepts (20/160): the judge passes some wrong
   answers. Caveat on the ground truth: cross-pairing pairs each case
   with its file neighbour, and the golden files are grouped by topic, so
   an adjacent "wrong" answer is often the same topic and can be partly
   applicable - some of these 20 are not clean errors. A topic-shuffled
   cross-pairing would tighten the negative set; that refinement, plus
   human-labeled correctness (the true gold standard here), is future
   work. The number is reported as-is rather than tuned to look better.

This is the capstone of the judge audit: the judge behind every score on
this project is now validated by chance-corrected agreement against known
truth, with the false-accept limit stated plainly.
