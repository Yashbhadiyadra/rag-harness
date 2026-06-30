# ADR-0004: Use LLM-as-judge for faithfulness and correctness metrics

## Status
Accepted

## Context
The evaluation layer scores each generated answer on three metrics:

| Metric | Type |
|---|---|
| Context Recall | Deterministic (set intersection) |
| Faithfulness | Semantic — requires understanding |
| Correctness | Semantic — requires understanding |

Faithfulness asks: "Is every claim in the answer supported by the retrieved context?"
Correctness asks: "Does the answer capture the key points of the reference answer?"

Neither can be computed with string matching or edit distance — they require
semantic understanding of natural language.

## Decision
Use `gpt-4o-mini` at `temperature=0` as an evaluation judge for faithfulness and
correctness. The judge receives a structured prompt with the question, context (or
reference answer), and the generated answer, and returns a score between 0.0 and 1.0.

The eval gate runs **nightly and on-demand only** — never on every PR. This is
enforced by keeping the nightly eval workflow separate from the per-PR CI workflow.

## Reasons
- **No viable non-LLM alternative.** ROUGE/BLEU scores are word-overlap metrics —
  they penalise valid paraphrases and reward surface-level matches. BERTScore is
  better but still does not reason about faithfulness (grounding in a specific context).
- **`temperature=0` for determinism.** Reproducibility is essential for a reliability
  system. The same answer should receive the same score across runs.
- **`gpt-4o-mini` for cost.** Faithfulness and correctness scoring are the most
  expensive operations in the pipeline. Using the cheapest capable model keeps the
  nightly eval affordable (estimated < $0.05 per full golden-set run at 30 cases).
- **Structured prompt constrains the judge.** Prompts explicitly ask for a decimal
  between 0.0 and 1.0 with no other output. The response is parsed and clamped.

## Tradeoffs
- **Evaluator bias.** An LLM evaluating LLM output may be systematically lenient
  (same training data, similar tendencies). Mitigated by keeping the golden set
  hand-verified and treating the judge score as a relative signal, not an absolute
  ground truth.
- **Cost.** Each eval run calls the LLM twice per golden case (faithfulness +
  correctness). At 30 cases this is 60 API calls. Acceptable at nightly frequency;
  unacceptable on every PR — hence the separate CI workflow.
- **Non-determinism risk.** Even at `temperature=0`, LLM outputs can vary across
  model versions. A model upgrade should trigger a manual eval review.

## Alternatives considered
- **ROUGE-L**: discarded — rewards word overlap, not semantic faithfulness.
- **Human evaluation**: gold standard but not automatable for a regression gate.
- **Dedicated eval frameworks (RAGAS, TruLens)**: considered but add heavyweight
  dependencies and obscure the scoring logic. Building our own keeps the eval layer
  transparent and reusable for Project 3.
