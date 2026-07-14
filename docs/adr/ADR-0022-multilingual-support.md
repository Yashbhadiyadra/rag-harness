# ADR-0022: Multilingual support (design)

Date: 2026-07-14
Status: Accepted (design only - no code in this ADR)

## Context

The RAG-evaluation surveys note that existing frameworks concentrate on
English and Chinese, leaving multilingual evaluation an open gap. This ADR
records what multilingual support would require, so the design is deliberate
rather than discovered later. No implementation lands here; it is scoped and
sequenced for when a concrete non-English use case exists.

## What already generalises

- **Corpus.** Bring-your-own-corpus (ADR-0019) already handles non-English
  docs: point `CORPUS_DOCS_SUBPATH` at a localised tree (e.g. the Kubernetes
  docs ship `content/zh/docs`, `content/fr/docs`, ...) and ingestion,
  chunking, provenance, and retrieval work unchanged. Embeddings from
  `text-embedding-3-small` are already multilingual.
- **Metrics and machinery.** Recall, precision, faithfulness, correctness,
  relevancy, the ablation runner, the gates, and the CIs are all
  language-agnostic - they operate on scores, not text.

## What multilingual support actually needs

1. **A per-language golden set.** The shipped golden set is
   English-Kubernetes-specific. Each language needs its own set, produced
   through the same generate-then-human-review pipeline pointed at the
   localised corpus. The reviewer must read that language.
2. **Judge validation per language.** LLM judges handle non-English text, but
   their reliability is not guaranteed to transfer. The judge audit
   (ADR-0014) should be re-run per language - calibration, near-gate noise,
   and discrimination can differ - before trusting the scores. This is the
   honest part: do not assume the judge is as reliable in French as in
   English; measure it.
3. **Judge-prompt localisation (optional).** The judge prompts are English
   and instruct the model to score. They can stay English (the model scores
   non-English answers fine) or be localised; the scale-sensitivity finding
   (ADR-0014) suggests testing rather than assuming either is equivalent.

## Decision

Treat multilingual support as a data-and-process extension, not a code
change: it is a new corpus (already supported) plus a new reviewed golden
set plus a re-run of the judge audit in that language. Nothing in the core
needs to change. Ship it when a real non-English corpus and a reviewer for
that language exist, not speculatively.

## Consequences

- The path is clear and cheap to start: ingest a localised corpus today; the
  work is the golden set and the per-language judge audit, both of which
  reuse existing tooling.
- The one non-obvious requirement recorded here is that judge reliability
  must be re-measured per language rather than assumed - consistent with the
  project's rule that nothing is trusted without a number.
