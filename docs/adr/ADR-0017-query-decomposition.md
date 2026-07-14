# ADR-0017: Query decomposition retrieval strategy

Date: 2026-07-14
Status: Accepted

## Context

A single dense query embeds to one point in vector space, so a multi-hop
question ("What is a StatefulSet and how does its Pod naming differ from
a Deployment?") is served poorly: one embedding cannot sit near the
StatefulSet passage and the Deployment-naming passage at once. NVIDIA's
enterprise RAG blueprint and the broader agentic-RAG trend treat query
decomposition as a standard retrieval stage; the strategy research
flagged multi-hop retrieval as a gap in this project (the golden set is
single-hop today).

## Decision

Add a `decompose` retrieval strategy: `DecompositionRetriever` asks the
LLM to split a question into the minimal set of standalone sub-queries
(capped at 4), retrieves the full top_k for each with a base retriever
(HybridRetriever), and fuses the ranked lists with Reciprocal Rank
Fusion (reusing the existing RRF used by hybrid). A single-intent
question decomposes to itself and delegates straight to the base
retriever, so the strategy is safe everywhere and only fans out when
there is genuinely more than one thing to retrieve. Decomposition
failures fall back to the raw query rather than erroring.

Like HyDE, this is a composable query-time transform. It is registered
in the factory and `_STRATEGY_ORDER`, so the ablation runner picks it up
automatically (now 12 configurations: 6 strategies x 2 corrective
modes).

## Consequences

- The harness can now serve multi-hop questions with evidence retrieved
  for each part, verified end-to-end on "What is a StatefulSet and how
  does its Pod naming differ from a Deployment?" - the answer cited
  passages found by different sub-queries.
- Cost: decomposition adds one LLM call per query plus one retrieval per
  sub-query. It is opt-in via strategy selection, not the default.
- The single-hop golden set does not exercise decomposition's advantage
  (most questions decompose to themselves), so a fair measurement needs
  multi-hop cases. Those are drafted through the existing golden
  expansion + human review pipeline (Phase 3 task 4), not hand-written,
  and the ablation is re-run once they land. Until then, decompose is
  expected to roughly match its base retriever on the single-hop set -
  the point is that it does not regress there while adding multi-hop
  capability.
