# ADR-0020: Closed-loop eval

Date: 2026-07-14
Status: Accepted

## Context

The golden set is built once, from generated candidates the owner reviews.
But the queries that matter most for reliability are the ones the deployed
system handles poorly in production, and those never fed back into the
golden set. Incumbents market this loop ("every failure becomes a test
case"); the project already has the exact machinery to close it - a
candidate queue and an interactive human review - so wiring production
failures into that queue is a small, high-value step.

## Decision

When a live `/query` produces a low-confidence answer, capture it as a
review candidate and append it to a candidate queue. Details:

- **Confidence signal**: a refusal answer is the cheap hot-path signal (no
  extra LLM call); an optional reference-free faithfulness score can tighten
  the signal for offline/batch capture.
- **Where it goes**: a separate queue file
  (`evals/review-queue/closed-loop.jsonl`, git-ignored working state), which
  the owner reviews with the existing tool
  (`rag_harness golden review --queue evals/review-queue/closed-loop.jsonl`).
  Captured candidates match the `GoldenCaseCandidate` schema, so no new
  review path is needed. Refusals are filed as `unanswerable`; answered
  low-confidence queries infer their topic from the top retrieved chunk.
- **Dedup**: a near-duplicate question already in the queue is not
  re-captured, so a repeated failing query does not flood the queue.
- **Safety**: the module never writes to the golden files. The human review
  gate before the golden set is inviolable - capture only proposes
  candidates. It is off by default (`CLOSED_LOOP_ENABLED=false`) and the
  hot-path hook is wrapped so a capture failure can never break a response.

## Consequences

- The golden set becomes a living system: production gaps flow into the same
  human-reviewed pipeline that built it, so coverage grows where the system
  actually struggles rather than only where the generator happened to sample.
- Opt-in and cheap: no extra LLM call, one file append on a low-confidence
  query, nothing on a confident one.
- Pairs with closed-loop capture and the golden review to make a full
  development loop: production trace -> candidate -> human review -> golden
  set -> gated eval.
- Follow-up (out of scope here): also capture gate-failing cases from the
  offline eval runner, and use the faithfulness signal in a sampled offline
  pass rather than only refusals on the hot path.
