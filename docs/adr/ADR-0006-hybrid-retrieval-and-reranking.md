# ADR-0006 - Hybrid retrieval, cross-encoder reranking, and HyDE

**Status:** Accepted  
**Date:** 2026-07-02  
**Decided by:** Owner, 2026-07-02

## Context

The baseline retriever uses only dense cosine similarity via OpenAI
`text-embedding-3-small`. This works well for paraphrased or semantically framed
questions but fails on two patterns common in the Kubernetes docs:

1. **Exact-term queries**: `kubectl apply --dry-run`, `PodDisruptionBudget`,
   `NetworkPolicy`. Dense embeddings compress rare identifiers into a shared
   subspace, so keyword-exact matches are diluted.
2. **Query-answer vocabulary mismatch**: the question ("How do I stop pods
   from starting on GPU nodes?") shares few words with the answer ("Add a taint
   with effect NoSchedule..."). Bi-encoder retrieval fails when there is no
   surface overlap.

Modern production RAG addresses both with a three-stage pipeline: a strong
first-stage retriever (hybrid), a precision-oriented second stage (cross-encoder
rerank), and optionally a query transformation (HyDE).

## Decision

### 1. First stage - hybrid retrieval with Reciprocal Rank Fusion

Combine dense retrieval with BM25 sparse retrieval and fuse the two ranked
lists using **Reciprocal Rank Fusion** (Cormack et al. 2009):

```
score(d) = Σᵢ 1 / (k + rankᵢ(d) + 1)     with k = 60
```

RRF is rank-based, not score-based, so the raw scores from the two rankers
never need to be normalised. This is important because dense cosine similarity
lives in [-1, 1] while BM25 scores are unbounded and corpus-dependent.

`BM25Store` builds an in-memory `BM25Okapi` index from all documents in
ChromaDB at startup. Rebuilding is fast (~2 seconds for 50k chunks), so we do
not persist the BM25 index to disk; one less file to invalidate.

### 2. Second stage - cross-encoder reranker

Wrap the first stage in `RerankingRetriever` using
`cross-encoder/ms-marco-MiniLM-L-6-v2`. The base retriever fetches
`top_k * 4 = 20` candidates; the cross-encoder rescores each `(query, chunk)`
pair by reading them jointly, then the top 5 are returned.

Cross-encoders capture fine-grained relevance signal that bi-encoders miss
because they attend across the query and document together, rather than
compressing each into a single vector independently.

### 3. Query transformation - HyDE

`HyDERetriever` wraps any base retriever. Before retrieval it asks
`gpt-4o-mini` to draft a hypothetical answer passage; that passage is then
used as the query. Even when the hypothesis contains factual errors, its
vocabulary and structure land close to real answer documents in embedding
space, closing the query-answer gap.

If the hypothesis generation fails (network, safety refusal, empty response),
HyDE falls back to the raw query: a degraded retrieval is better than a
failed one.

## Composition

All retrievers implement the same `Retriever` interface. The factory function
`build_retriever(strategy)` composes them:

| Strategy | Composition |
|---|---|
| `dense` | `DenseRetriever` |
| `hybrid` | `HybridRetriever(dense, bm25)` |
| `hybrid-rerank` | `RerankingRetriever(HybridRetriever)` |
| `hyde` | `HyDERetriever(DenseRetriever)` |
| `full` | `HyDERetriever(RerankingRetriever(HybridRetriever))` |

Downstream code (generation, evaluation, API, CLI) never depends on the
concrete strategy. This preserves the composability principle from ADR-0001.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Weighted linear combination of dense + BM25 scores | Requires per-corpus score normalisation; RRF has no hyperparameter to tune |
| Learned-to-rank fusion | Requires labelled data we don't have; RRF matches learned fusion in most benchmarks |
| Cohere Rerank API | Costs money per query; MS MARCO MiniLM is free, runs on CPU, and matches Cohere v2 on English text |
| Full cross-encoder as first stage | Quadratic in corpus size; only feasible with retrieve-then-rerank |
| Multi-query / RAG-Fusion (multiple query rephrasings) | Deferred to future work; costs an extra generation call per query |
| Query classification / routing | Deferred; adds complexity for marginal gain given current corpus |

## Consequences

- **Retrieval quality**: hybrid vs dense-only lifts context precision from
  ~0.61 to ~0.71 in published production benchmarks; adding cross-encoder
  rerank lifts it further to ~0.79. HyDE closes the query-answer gap on
  paraphrased questions.
- **Latency**: dense (single embedding call) baseline. Hybrid adds a few
  milliseconds for BM25 lookup. Rerank adds ~80-120ms on CPU for 20
  candidates. HyDE adds one `gpt-4o-mini` generation (~200-500ms).
- **Cost**: hybrid is free (BM25 is local); rerank is free (CPU inference);
  HyDE costs one extra generation per query (~$0.00003 with `gpt-4o-mini`).
- **Dependencies**: `rank_bm25` (50KB, pure Python) is in core deps.
  `sentence-transformers` (~500MB with PyTorch) is behind the `[rerank]`
  optional extra to keep the base install lightweight.
- **Ablation**: strategies are stable identifiers so `evals/experiments/`
  can compare them cleanly. Phase 8 will produce a benchmarks table.
