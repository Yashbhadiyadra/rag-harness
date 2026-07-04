# ADR-0007 — Corrective RAG: critic-and-retry loop

**Status:** Accepted
**Date:** 2026-07-02

## Context

The baseline pipeline (retrieve → generate) is passive: it produces an answer
from whatever chunks the retriever returned, without checking whether those
chunks are actually relevant. This has two failure modes:

1. **Silent hallucination on irrelevant retrieval.** When the retriever
   fetches off-topic chunks, the generator still produces confident-sounding
   text — either extrapolating from weak signal or reverting to the "not
   enough information" fallback. There is no signal telling us which one
   happened.
2. **No self-correction.** A user query with unusual phrasing may miss the
   right documents entirely. The pipeline cannot recover; it has no way to
   try again with a different query.

Corrective RAG (Yan et al. 2024, arXiv:2401.15884) introduces an active
retrieval evaluator that scores each chunk for relevance and routes the
pipeline into one of three branches. This ADR describes our adaptation.

## Decision

### 1. Relevance critic (single-call batch scoring)

Add `RelevanceCritic` in `generation/critic.py`. On each query, the critic
sends the query and all retrieved chunks to `gpt-4o-mini` in a **single**
call with `response_format={"type": "json_object"}`. The response is a
`{"scores": [0.9, 0.4, ...]}` object with one score per chunk in the same
order.

Single-call batch scoring is deliberate — the alternative (one API call per
chunk) is N times more expensive. The critic's system prompt includes a
scoring rubric so scores are calibrated: 1.0 = directly answers, 0.7 =
mostly answers, 0.4 = related, 0.1 = off-topic, 0.0 = irrelevant.

### 2. Three-way categorisation

Scores collapse to a routing decision:

| Category | Rule | Action |
|---|---|---|
| Correct | `max(scores) ≥ 0.7` | Filter out chunks below 0.3; generate answer |
| Ambiguous | `0.3 ≤ max(scores) < 0.7` | Filter out chunks below 0.3; generate anyway |
| Incorrect | `max(scores) < 0.3` | Reformulate query and retry; on final failure, refuse |

Thresholds (0.7 and 0.3) are configurable and were chosen to align with the
critic's scoring rubric — a chunk scored 0.7+ is claimed to be "mostly
answers the question" per the rubric, so promoting the batch to Correct is
safe. A batch whose best chunk is below 0.3 is claimed to be off-topic, so
generating on it would be hallucination.

### 3. Reformulation and retry

When categorised as Incorrect, ask `gpt-4o-mini` to rewrite the query to
"surface different keywords, terminology, or synonyms" while preserving
intent. Retrieve again with the rewritten query, re-score, re-categorise.

The retry cap is 1 by default (one reformulation attempt), giving up to
two total retrieval attempts. If both are Incorrect, return the standard
`"I do not have enough information..."` refusal — byte-identical to the
generator's own fallback so evaluation sees a consistent signal.

### 4. Critic scores against the original query, always

Even after reformulation, the critic scores chunks against the **original**
user query, not the reformulated one. Rationale: the reformulation is a
search-side tool for surfacing different keywords, not a semantic change.
The relevance judgement must stay anchored to what the user actually asked
so scores are comparable across attempts.

### 5. Composability

`corrective_generate(query, retriever, ...)` accepts any `Retriever` — dense,
hybrid, hybrid-rerank, hyde, or the full pipeline. Corrective RAG composes
over the strategies from Phase 5 rather than replacing them.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Per-chunk critic calls (one API call each) | N times more expensive; batch call with structured output achieves the same signal for a single call |
| Web-search fallback (as in CRAG paper) | Our corpus is deliberately bounded to the pinned K8s docs commit; introducing external web results would break provenance guarantees |
| Sentence-level knowledge refinement (decompose-score-recompose) | Adds a second LLM pass per chunk for marginal precision gain; defer to a future phase if measurements justify |
| Unbounded retries | Cost escalates fast; a single retry doubles the maximum-cost path but bounds worst case; interviewers will ask about cost controls |
| Silent fallback to non-corrective on critic failure | Better to fail loudly (zero scores → Incorrect → refuse) than pretend the critic worked; refusal preserves the faithfulness invariant |

## Consequences

**Quality:**
- Eliminates silent hallucination when retrieval misses. The refusal message
  is byte-identical to the generator's own fallback, so faithfulness/
  correctness scoring stays consistent.
- Recovers some queries that would otherwise miss via query reformulation.
- Filters weak chunks out of the generation context, tightening faithfulness.

**Cost:**
- Best case (all Correct): +1 critic call vs baseline. Roughly 1.5× cost.
- Retry case (Incorrect → Correct): +2 critic calls + 1 reformulation +
  1 extra retrieval. Roughly 3-4× cost.
- Worst case (Incorrect → Incorrect → refuse): +2 critic calls + 1
  reformulation + 1 extra retrieval, then no generation. Also 3-4× cost.

Gated behind `CORRECTIVE_RAG_ENABLED=false` by default. Users opt in.

**Telemetry:**
- `CorrectiveResult` returns not just the answer but the category, attempt
  count, per-chunk scores, and the reformulated query if used. Phase 7
  (observability) and Phase 8 (ablation study) will use this to attribute
  wins and losses to specific corrective branches.

## Amendments

**2026-07-04 — Latency observation from the Phase 8 ablation.** Stacking the
corrective loop on top of the most expensive retrieval strategy (`full` =
HyDE + Rerank + Hybrid) roughly **doubles the p50 latency**: `full` baseline
p50 is 10.5s per case; `full` + corrective p50 is 18.8s per case (p95 grows
from 17.6s to 24.0s). See `evals/experiments/ablation_20260704T092511+0000_e148311.md`.

Combined with the ablation's finding that corrective showed no consistent
improvement across strategies at n=30, this argues against enabling corrective
on top of `full` for latency-sensitive deployments. Corrective is best kept
as an opt-in feature for cases where retrieval is expected to be weak — a
degradation-recovery tool, not a default enhancement to the strongest
pipeline. Larger golden sets in future phases may revise this stance.
