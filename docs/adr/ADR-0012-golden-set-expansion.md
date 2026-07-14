# ADR-0012 - Golden-set expansion to n≈100 with unanswerable and version-sensitive categories

**Status:** Accepted  
**Date:** 2026-07-07  
**Decided by:** Owner, 2026-07-07

## Context

The eval golden set is 30 hand-verified cases. At n=30 every headline
number carries a bootstrap CI wider than most inter-strategy deltas,
so the ablation is not statistically powered to distinguish strategies
that are actually close (ADR-0011). Two specific failure modes are also
invisible at the current n:

- **Confident hallucination on out-of-corpus questions.** The demo lets a
  visitor ask anything, and the honest failure mode is a refusal.
  Without unanswerable cases in the eval set the refusal path is never
  scored. A silent regression that starts confabulating answers is only
  visible on the live demo.
- **Cross-version misinformation.** The corpus is pinned to K8s v1.32
  (ADR-0002). When a chunk documents version-dependent behavior (e.g.,
  `apps/v1beta1` removal, PodSecurityPolicy → Pod Security Admission)
  the LLM may answer using its pretraining knowledge of newer K8s
  versions instead of the pinned corpus. Without version-sensitive
  cases the metric can't catch this.

Constraints:

- The golden set is reviewed like code (ADR-0004 §"never auto-generated
  without review"). Any expansion mechanism must funnel through explicit
  human review.
- The corpus is pinned to an immutable git SHA (ADR-0002); any
  version-sensitive reference answer must be verifiable against the
  actual chunk text, not against the LLM's general knowledge.
- Bootstrap CIs (ADR-0011) will be recomputed on the expanded set,
  which changes every published headline number in the ablation table.

## Decision

Grow the golden set to n≈100 in three categories, all generation-
assisted then hand-reviewed:

| Category | Count target | Purpose |
|---|---|---|
| **topic** (grows existing files) | 80 candidates → ~50 accepted | Expand coverage across workloads / networking / storage / scheduling / cluster / rbac |
| **unanswerable** (new file `evals/golden/unanswerable.json`) | 40 candidates → ~20 accepted | Score the refusal path. Correct behavior is the honest "I don't have enough information" refusal. |
| **version-sensitive** (new file `evals/golden/version-sensitive.json`) | 40 candidates → ~20 accepted | Score cross-version misinformation. Correct behavior is the answer derivable from the pinned chunk. |
| **Total** | 160 candidates → ~100 accepted | |

### Outcome (review closed 2026-07-14)

The generator's drop gates produced 143 candidates (fewer than the 160
target, mainly in the filtered unanswerable and version-sensitive
categories). The owner reviewed all 143 by hand; 130 were accepted and
13 skipped. The golden set grew from 30 to 160 cases.

| Category | Candidates | Accepted | Skipped |
|---|---|---|---|
| topic (cluster/networking/rbac/scheduling/storage/workloads) | 80 | 71 | 9 |
| unanswerable | 24 | 21 | 3 |
| version-sensitive | 39 | 38 | 1 |
| **Total** | **143** | **130** | **13** |

Skips fell into four defensible buckets: mislabeled unanswerables the
pinned corpus actually documents (3), degenerate answers that were only
unrendered doc-template markup or a tautology (4), navigation questions
with link-only answers (5), and one weak standalone question. Every
accepted case was human-reviewed and its answer confirmed grounded in
the cited source chunk. Landed in commit 164ff8c.

### Two-stage pipeline

**Stage 1 - candidate generation** (`scripts/expand_golden_set.py`,
this ADR):

- Sample chunks from the ingested ChromaDB collection.
- For **topic** candidates: prompt `gpt-4o-mini` to draft one question
  that the sampled chunk directly answers, plus a reference answer
  written from the chunk content only.
- For **unanswerable** candidates: prompt the LLM to draft plausible
  K8s questions that the pinned v1.32 corpus does NOT answer, then
  run the drafted question through the actual retriever. Candidates
  whose top hit exceeds a similarity threshold (0.85; raised from an
  initial 0.75 after the pilot showed 0.75 silently dropped genuinely
  unanswerable questions that were merely topically related to real
  chunks) get dropped: those are questions the corpus does answer.
  Surviving candidates
  carry `retrieval_evidence` (top-k hits + similarities + an
  LLM-written explanation of why the hits don't answer the question)
  so the reviewer confirms the classification, not just trusts the
  classifier.
- For **version-sensitive** candidates: sample chunks that match a
  regex for version-referencing text (`apps/v1beta`, `removed in`,
  `deprecated in`, `since v1.*`, `graduated in`, etc.), prompt the
  LLM to draft a version-dependent question with a reference answer
  derived from the chunk.

Output is a JSONL review queue at `evals/review-queue/candidates.jsonl`.
This file is not tracked in git (working state, not artefact).

**Stage 2 - human review** (`python -m rag_harness golden review`,
D-a-2):

The reviewer walks the queue one candidate at a time. For each:

- The **source chunk text is surfaced prominently** so the reviewer
  verifies the draft reference answer against ground truth, not the
  LLM's suggestion. This matters most for version-sensitive candidates
  where the LLM may answer from a newer K8s version than the pin.
- For **unanswerable candidates the retrieval evidence is displayed
  in full**: top-k hits, similarities, and the LLM-written reason. The
  reviewer confirms the classification is correct (truly not-in-corpus)
  rather than trusting that low similarity means "not answered."
- The reviewer answers `[y]es include / [n]o skip / [e]dit answer /
  [q]uit`. On accept, the case gets a real ID (e.g.,
  `unanswerable-001`) and lands in the appropriate
  `evals/golden/<category>.json`.

The queue is idempotent - quitting mid-review saves state and resuming
skips already-decided rows.

### Provenance

Every accepted case is recorded in the golden JSON with the same
`GoldenCase` schema as before. Provenance of each case is tracked in
this ADR (below), not in the JSON, so the schema does not change:

| ID range | Origin | Reviewer |
|---|---|---|
| `cluster-001` … `cluster-005` | Hand-written | Owner |
| `networking-001` … `networking-006` | Hand-written | Owner |
| `rbac-001` | Hand-written | Owner |
| `scheduling-001` … `scheduling-005` | Hand-written | Owner |
| `storage-001` … `storage-005` | Hand-written | Owner |
| `workloads-001` … `workloads-008` | Hand-written | Owner |
| Growth of the above (added via the Stage 2 review after this ADR) | Generation-assisted, hand-reviewed | Owner |
| `unanswerable-*` | Generation-assisted, hand-reviewed | Owner |
| `version-sensitive-*` | Generation-assisted, hand-reviewed | Owner |

This table gets updated when the Stage 2 review closes. A reviewer
looking at the golden set can trust that every case was seen by a
human, and the ADR tells them which cases were originally drafted by
a human vs. drafted by an LLM and then approved.

## Alternatives considered

| Option | Why rejected |
|---|---|
| **Hand-write all 100 cases from scratch** | ~10× more review time; no useful signal added over the review-approve loop we're doing anyway. Generation-assisted-then-hand-reviewed is the same rigour with less time. |
| **Auto-generate and skip human review** | Violates ADR-0004's explicit "never auto-generated without review" discipline. LLM-drafted questions have distinct failure modes (leading, ambiguous, or answerable from outside the chunk) that a reviewer catches. |
| **Score the refusal path via a separate telemetry-only check** | Would only catch refusal-vs-not, not refusal-quality. Unanswerable golden cases with reference answers let the same LLM-judge suite score refusal exactly the same way it scores answers. |
| **Retrieval similarity as the sole gate for unanswerable classification** | Similarity below a threshold does NOT prove not-in-corpus; a semantically-unusual question could match nothing while still being answerable by a chunk with different wording. Surface the evidence and let the human decide. |
| **Store candidate provenance in each `GoldenCase` JSON** | Adds a schema field used only by this ADR. Keeping provenance in the ADR keeps the eval-time schema minimal. |

## Consequences

- The reliability gate thresholds (in `config.py`) will need re-
  calibration once the ablation is re-run on n=100. Deltas that were
  significant at n=30 may or may not survive the tighter CIs; some new
  deltas will emerge as significant.
- The public metrics page will show real bootstrap CIs (ADR-0011)
  instead of the current `(no CI - pre-bootstrap run)` labels once the
  first ablation on the expanded set lands.
- Unanswerable cases will surface any silent regression in the refusal
  path. The `answer_relevancy × correctness` divergence check
  (relevant-but-incorrect) becomes materially more meaningful.
- Version-sensitive cases discipline the correctness judge: an answer
  that is "correct in K8s v1.34" but not in v1.32 must score low, and
  the pinned reference answer makes that unambiguous.
- One-time cost: ~$0.20 in OpenAI drafting + ~2–3 hours of Owner
  review time. Recurring cost is zero - the expanded golden set is
  static.

## Implementation shape (informational)

- **Commit D-a-1** (this commit): candidate generator in
  `scripts/expand_golden_set.py` + tests + this ADR. No golden-set
  file is modified.
- **Commit D-a-2**: `python -m rag_harness golden review` CLI + tests.
- **Runtime (Owner)**: `python -m scripts.expand_golden_set` writes
  the queue; `python -m rag_harness golden review` walks it.
- **Commit D-a-3**: accepted candidates land in
  `evals/golden/<category>.json`. Provenance table above is updated
  with actual accepted-count ranges.
- **Commit D-a-4**: re-run ablation. `evals/history/runs.jsonl` grows.
  Metrics page shows real CIs.
- **Commit D-a-5** (optional follow-up): recalibrate threshold config
  from the new baseline.
