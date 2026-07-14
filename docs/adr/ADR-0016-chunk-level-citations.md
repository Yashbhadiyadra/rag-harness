# ADR-0016: Chunk-level inline citations

Date: 2026-07-14
Status: Accepted

## Context

The `/query` response listed `sources` (the files behind an answer) but
did not tie any specific claim to the chunk it came from. Per-passage
citation is the industry table-stakes for grounded RAG (NVIDIA NeMo
Retriever ships citation-level attribution), and it is also a
reliability lever: an answer that must cite its source per claim is
forced to stay grounded.

## Decision

Number the context passages `[1], [2], ...` (already done in
`_build_context`) and instruct the generator to cite the passages it
used inline with those markers. The `/query` response gains a
`citations` list mapping each cited marker to the exact chunk
(`source_file`, `heading_path`). `sources` is unchanged and `citations`
defaults to empty, so existing clients are unaffected. Marker parsing
(`citations.py`) is pure and network-free; a marker that points past the
chunk list is skipped rather than raising, so a stray citation can never
break a response.

This changes the generation prompt, so per the governing rule the eval
delta was measured before landing.

## Consequences

Quality delta (golden eval, dense strategy, cache on):

| Metric | Before | After |
|---|---|---|
| Context recall | 0.767 | 0.767 |
| Context precision | 0.915 | 0.915 |
| Faithfulness | 0.900 | **0.930** |
| Correctness | 0.755 | **0.792** |
| Answer relevancy | 0.833 | **0.867** |

Citations did not just add attribution - they improved answer quality
across every judged metric (correctness +0.037, faithfulness +0.030,
relevancy +0.034). Requiring the model to attribute each claim to a
numbered passage keeps it grounded in the retrieved content, the same
mechanism that helps the injection hardening (ADR-0015). Retrieval
metrics are unchanged because retrieval was untouched.

This is the third prompt change in a row (ADR-0015 hardening, then
citations) where a reliability-motivated instruction also raised
quality, which is the recurring lesson: grounding constraints and answer
quality are aligned, not in tension.

The citation markers now appear in generated answers, including the
answers the judge scores. That is intentional and measured here; a
follow-up (Phase 3 task 2) adds a citation-accuracy metric that checks
each cited passage actually supports the claim attached to it.
