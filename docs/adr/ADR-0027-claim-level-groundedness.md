# ADR-0027: Claim-level groundedness (four-way typology)

Date: 2026-08-07
Status: Accepted

## Context

Faithfulness today is a single holistic judge call: given the question,
context, and answer, the judge returns one score in `[0,1]` meant to
represent "the fraction of claims that are supported" (`_FAITHFULNESS_PROMPT`
in `evaluation/metrics.py`). That number gates releases and is a headline
reliability metric.

Two limits matter for a product whose thesis is provable trust:

1. It is one opaque number. A 0.7 could mean "most claims supported, one
   unsupported" or "everything half-supported", and it cannot tell an
   unsupported-but-harmless claim from a claim that directly contradicts the
   source. For a trust layer, the difference between "not in the context" and
   "conflicts with the context" is the difference between a gap and a lie.
2. The 2026 state of the art moved past binary support. Grounding-evaluation
   work (GSAR and related span-level detectors) classifies each claim into a
   typology rather than scoring the answer as a whole, which both improves
   detection and makes the result explainable per claim.

The governing rule is that no capability lands unmeasured, and the gate must
stay stable while we evaluate a new scorer.

## Decision

Add a claim-level groundedness scorer as an **additive** probe. It does not
replace the holistic faithfulness judge or the release gate; it runs alongside
so the two can be compared on the golden set before any gate change.

The scorer makes one structured judge call per answer that (a) extracts the
atomic claims in the answer and (b) classifies each claim against the retrieved
context into one of four types:

- **grounded** - directly supported by the context.
- **ungrounded** - not addressed by the context (a gap, not a conflict).
- **contradicted** - conflicts with the context (the dangerous type).
- **complementary** - reasonable, non-conflicting information that extends
  beyond the context (e.g. common-knowledge scaffolding).

From the per-claim labels it derives:

- `groundedness = grounded / total_claims` - the strict headline score.
- `hallucination_rate = (ungrounded + contradicted) / total_claims`, with
  `contradicted` surfaced separately because a contradiction is worse than a
  gap.

One structured call per answer keeps cost comparable to the current single
faithfulness call. The scorer lives in `evaluation/claim_eval.py` with a
sync facade and a Markdown report, mirroring the other probe modules
(`citation_eval`, `abstention_eval`), and is exposed as `rag-harness claim-eval`.
Parsing is defensive: a malformed judge reply degrades to zero claims rather
than raising, so the probe can never break a run.

Swapping the release gate from holistic faithfulness to claim-level
groundedness is explicitly **out of scope here** and gated on the measured
agreement below. The claim-level verify-and-strip pass in the corrective loop
is future work, gated on the same evidence.

## Consequences

Measured on a 30-case golden sample (dense strategy, judge `gpt-4o-mini`,
cache on):

| Metric | Value |
|---|---|
| Cases | 30 |
| Total atomic claims | 122 |
| Grounded | 118 |
| Complementary | 4 |
| Ungrounded | 0 |
| Contradicted | 0 |
| Claim-level groundedness | 0.967 |
| Holistic faithfulness (same cases) | 0.933 |
| Mean absolute difference | 0.046 |

Two things the numbers establish. First, the scorers agree closely: mean
absolute difference of 0.046 per case, which is the evidence needed to trust the
claim-level view as a drop-in measurement rather than a different metric.
Second, on this sample the answers were cleanly grounded, zero ungrounded and
zero contradicted claims, with four complementary claims (reasonable background
that goes beyond the passages without conflicting). This is where the two
scorers diverge and why claim-level is slightly higher (0.967 vs 0.933): the
holistic judge has no way to credit a complementary claim as non-hallucinated,
so it quietly penalises correct background as if it were partial support. The
claim view separates "beyond the context but fine" from "not supported."

The durable win is not this sample's clean result but the vocabulary. When an
answer does contain an ungrounded or contradicted claim, this scorer names which
claim and which type; a single 0.9 cannot. That per-claim, per-type visibility
is the reason to prefer it in the Reliability Audit and the writeup, and the
reason a future gate change is worth measuring.

Because the scorer is additive, the release gate and its history are unchanged.
Promoting claim-level groundedness to the gate, and adding a corrective
verify-and-strip pass, are follow-ups that will each carry their own measured
delta.
