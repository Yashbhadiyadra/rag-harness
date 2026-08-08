# ADR-0030: Open-model groundedness detector

Date: 2026-08-07
Status: Accepted

## Context

Claim-level groundedness (ADR-0027) uses an LLM judge: one API call per answer
to classify each claim. It is accurate and gives the four-way typology, but it
costs money per call, which matters for the hosted audit at volume. The 2026
research direction (compact detectors such as Luna and LettuceDetect) shows that
a small encoder can localize unsupported content at a fraction of an LLM judge's
cost. A trust layer should be model-agnostic and cost-aware, and "a cheap open
judge can rival a frontier one" is a genuine product edge worth having a real
number for.

The codebase already declares transformers, torch, and sentence-transformers in
the `[rerank]` extra, so an open detector can be added with no new dependency.

## Decision

Add an open-model groundedness detector (`evaluation/open_detector.py`) that runs
a small NLI cross-encoder locally at zero API cost. Each answer sentence is a
hypothesis tested against the retrieved context: entailment maps to grounded,
contradiction to contradicted, neutral to ungrounded. A claim entailed by any
chunk is grounded; otherwise contradicted by any chunk is contradicted;
otherwise ungrounded.

It is additive and optional, selected with `claim-eval --detector open` (default
stays `llm`). The model (`open_detector_model`, default
`cross-encoder/nli-MiniLM2-L6-H768`) is lazy-loaded and cached; if the rerank
extra is absent the detector raises with install guidance. NLI has no
complementary class, so a reasonable-but-beyond-context claim is labelled
ungrounded, the conservative choice; the LLM judge remains the scorer that can
draw that finer distinction. This is the cheap open second opinion, not a
replacement.

Tests mock the model so CI never downloads weights.

## Consequences

Measured on the first 30 golden cases (dense strategy), open detector
(`cross-encoder/nli-MiniLM2-L6-H768`, local CPU) against the LLM claim judge
(`gpt-4o-mini`) on the same generated answers:

| Metric | Open NLI detector | LLM judge (ADR-0027) |
|---|---|---|
| Claims / sentences | 101 | 122 |
| Groundedness | 0.366 | 0.967 |
| Ungrounded + contradicted | 64 (27 + 37) | 1 |

**This is a negative result, and it is the point of measuring.** The small
off-the-shelf NLI detector does not rival the LLM judge: it labels 64 of 101
grounded sentences as ungrounded or contradicted on answers the validated judge
scores at 0.967 and the reliability gate passes at 0.94 faithfulness. Sentence-
level NLI is not RAG groundedness. A correct answer sentence paraphrases,
combines several passages, or uses different wording than any single chunk, so a
strict entailment model returns neutral or even contradiction. A small model
makes this worse. The "a cheap open detector can rival a frontier judge" thesis
does not hold with an off-the-shelf NLI model on this task.

**Decision revised by the evidence:** the detector ships as an **experimental,
off-by-default option** (`claim-eval --detector open`), documented as not
reliable, kept only because the code is the reproduction of this finding. It is
not wired into the audit or the gate, and must not be, until a fit-for-purpose
detector clears the bar. The real follow-ups are a purpose-built model
(LettuceDetect-style, or an NLI model fine-tuned on the golden set) and, only if
one earns its groundedness number, a cheap pre-filter that escalates uncertain
claims to the LLM judge. Until then the LLM judge remains the only groundedness
scorer the product trusts.
