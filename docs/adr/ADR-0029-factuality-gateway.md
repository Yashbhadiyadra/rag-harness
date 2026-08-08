# ADR-0029: Factuality gateway (claim-level verify-and-regenerate)

Date: 2026-08-07
Status: Accepted

## Context

ADR-0027 added a claim-level groundedness scorer that labels each atomic claim
in an answer as grounded, ungrounded, contradicted, or complementary. It
measures; it does not act. The scorer explicitly left a verify-and-strip pass as
future work.

The 2026 production pattern (a "factuality gateway") closes that loop: after an
answer is drafted, a cheap pass checks each claim against the retrieved context
and removes or regenerates the unsupported ones before the answer reaches the
user. A reported deployment cut hallucination-tainted outputs from 9% to 2.2%
with false positives under 0.5% at a few hundred milliseconds of added latency.
This is directly on-thesis: the trust layer should not only *report* that an
answer is grounded, it should *enforce* it.

## Decision

Add a factuality gateway that runs after generation inside the corrective loop,
gated by `factuality_gateway_enabled` (default off). It is off by default for two
honest reasons: it adds LLM calls, and on the already-grounded pinned demo
corpus there is usually nothing to strip (ADR-0027 measured zero ungrounded and
zero contradicted claims on a clean sample). It earns its keep on low-trust or
noisy corpora, which is exactly the hosted multi-tenant case.

The gateway (`generation/factuality.py`) reuses `classify_claims` (ADR-0027):

1. Classify the drafted answer's claims against the chunks actually used.
2. If no claim is ungrounded or contradicted, return the answer unchanged.
3. Otherwise regenerate once, feeding the flagged claims back with an
   instruction to answer using only context-supported content and to omit the
   unsupported claims. Regeneration (rather than blind sentence deletion) keeps
   the prose and citations coherent; when nothing is supported it collapses to
   the standard refusal, which is the correct outcome.

It regenerates at most once to bound cost and latency. The corrective loop
records whether the gateway revised the answer (`factuality_revised`) so the
behaviour is observable. The default pipeline and the release gate are unchanged
because the flag is off.

## Consequences

Two measurements, judge `gpt-4o-mini`, cache on. The honest result is a null on
realistic inputs and a verified mechanism on a crafted one.

**1. On realistic inputs the gateway does not fire.** On the 8 out-of-corpus
questions over real-but-irrelevant context (the abstention harness, where a weak
system would improvise), the base pipeline already refused all 8 and produced
zero ungrounded or contradicted claims, so the gateway had nothing to change:

| Metric | Gateway off | Gateway on |
|---|---|---|
| Questions | 8 | 8 |
| Answers with an ungrounded/contradicted claim | 0 | 0 |
| Correct refusals (abstention) | 8/8 | 8/8 |

This matches ADR-0027's finding that the pinned corpus plus the strict
context-only prompt already keeps the base system well-grounded. On this corpus
the gateway is pure added cost.

**2. The mechanism is verified end to end.** Given a drafted answer that does
contain a fabricated claim ("Pods automatically scale to zero when idle",
unsupported by the context), the real gateway flagged 1 of 2 claims and
regenerated an answer that dropped the fabrication and kept the grounded claim.
So the code works; it simply has nothing to correct on a corpus this clean.

**Verdict.** The gateway ships off by default. On the current well-grounded
corpus it is unjustified (measured null benefit, real cost), which is the
governing rule doing its job: the feature does not get to claim value it cannot
show. Its purpose is insurance for the low-trust and noisy tenant corpora the
hosted product will face, where the base system will not be this clean. That
value is a hypothesis, not yet a measurement, and is gated on a hosted low-trust
corpus providing a real test bed. A strip-without-regenerate fast path and
per-tenant enablement are follow-ups gated on the same evidence.
