# ADR-0010 — Cloud Run + scale-to-zero + baked Chroma index

**Status:** Accepted
**Date:** 2026-07-05
**Decided by:** Owner, 2026-07-05

## Decision

Deploy the RAG harness to **Google Cloud Run**, scale-to-zero, `max-instances=1`,
`concurrency=40`. Bake the 83 MB `chroma_db/` snapshot **into the container
image** as an immutable, versioned artefact. Enforce a **global daily request
cap** and a **per-IP rate limit** in-process, plus a Cloud Billing **$10/month
budget ceiling** with 50%/90%/100% alerts. Provide a `DEMO_ENABLED=false`
kill switch env var.

At demo scale this arrangement pays only when someone queries, cannot burn
more than a bounded amount per day, and requires no external counter store
(Redis, Firestore) to enforce its cost ceiling.

## Context

Roadmap Phase 10 turns the harness into a public demo a stranger can hit at
a URL. Two constraints dominate every other decision:

1. **Cost cannot surprise.** This is a public demo attached to an OpenAI
   key. A single scraper or misbehaving bot must not be able to run up a
   bill. Guardrails must be in from the first live commit, not bolted on.
2. **Cold-start latency must stay bearable.** Scale-to-zero is only
   defensible if the first query after an idle period returns in a
   reasonable time. The query path already pays a network round-trip to the
   OpenAI embeddings API — anything we add on top of that erodes the
   experience.

Three decisions are bundled in this ADR because they are entangled: the
hosting choice constrains persistence, and the persistence choice constrains
whether an in-memory global counter is safe.

## Hosting choice: Cloud Run

### Options

| Option | Scale-to-zero | Base monthly $ | Cold-start | Notes |
|---|---|---|---|---|
| **Cloud Run (chosen)** | Yes | $0 | 2–4 s cold-start once Chroma is baked in | GCP familiarity; native Secret Manager integration; Workload Identity Federation for GitHub Actions; free tier is generous |
| Fly.io | Yes (via `auto_stop_machines`) | $0 | 3–6 s cold-start | Simpler Docker Compose story if we needed Postgres; less useful here since Phoenix + SQLite is our tracing choice (ADR-0009) |
| Render | Free tier does not scale to zero, spins down after 15 min then cold starts in ~30 s | $0 free / $7 paid | 30 s+ | Free-tier cold-start is a demo killer |
| Self-managed VM | No | $5–10 minimum | 0 s (always warm) | Loses the point — pays 24/7 for something used a handful of times per day |

Cloud Run wins on operational fit and the GCP integrations we already
touch elsewhere (Secret Manager, Cloud Billing budgets, Artifact Registry).
Fly.io was a viable second choice; the tiebreaker was the tracing backend
decision in ADR-0009 (Phoenix + SQLite) which removed Fly's Postgres
advantage.

### Rejected alternative — never scale to zero

Setting `min-instances=1` avoids the cold-start entirely at the cost of
~$5/month for an idle container. For a portfolio demo that will see spiky,
low-total traffic (recruiters clicking through in bursts, then nothing for
days), scale-to-zero is the correct default. If cold-start ever becomes
demoable-in-90-seconds hostile, we revisit this — not before.

## Persistence choice: bake Chroma into the image

The Chroma index is 83 MB on disk. The corpus is pinned to an immutable git
SHA (ADR-0002). Rebuild cadence is "when someone bumps the corpus," which
is an ADR-level event, not a background job.

### Options

| Option | Cold-start added latency | Monthly $ | Complexity | Reproducibility |
|---|---|---|---|---|
| **Bake into image (chosen)** | ~2–4 s (Chroma opens 83 MB SQLite + HNSW dir) | $0 (Artifact Registry has generous free tier) | Zero new infra | Image tag == corpus SHA — atomic rollback |
| Rebuild on boot | **10–30 min** + ~$0.05–0.20 OpenAI embedding spend per cold start | $0, but scale-to-zero becomes hostile | Ingest path must run inside the runtime container | Deterministic if corpus SHA stable, but slow |
| Mounted storage (Filestore) | 5–15 s FUSE mount + Chroma open | Filestore basic is ~$230/month minimum | New failure mode, IAM plumbing | Same as baked, minus atomic swap |
| Mounted storage (GCS FUSE via Cloud Run second-gen) | 5–10 s mount + Chroma open | Pennies per month | New failure mode, larger container CPU footprint | Same as baked |

Baking wins for a reason that is easy to miss on paper: **query-time
embedding still calls OpenAI**. Whatever we do at the storage layer, the
first-query wall clock is dominated by the round-trip to the embeddings
API. Shaving Chroma load time is the only cold-start lever we control, and
baking wins that outright.

Concrete change (in a follow-up commit): CI builds the image with
`chroma_db/` present. The runtime container reads from `/app/chroma_db/`.
`VOLUME` declarations in the Dockerfile are dropped. The embedding cache
(`embedding_cache.db`, 208 MB, dev-only) is **not** copied — production
reads from the index, not the cache. The cache is a build-time speedup for
`make ingest`, nothing more.

### Rejected alternative — sentence-transformers for local embedding

If we ran the embedding model locally (e.g. `all-MiniLM-L6-v2`), we'd
eliminate the OpenAI round-trip in the query path and the cold-start
argument for baking becomes stronger still. But local embedding would break
compatibility with the index built by the OpenAI `text-embedding-3-small`
model — a re-embedding of the whole corpus and a new eval baseline. That is
a project-3 concern (ADR-0002 pinning applies to embeddings too), out of
scope for Phase 10.

## Cost guardrail architecture: single-instance + in-memory counter

The global daily request cap must be enforced across all in-flight requests.
Two implementations:

| Option | Cost | Complexity | Correctness under scale |
|---|---|---|---|
| **max-instances=1 + in-memory counter (chosen)** | $0 | Zero — a Python object with a lock | Perfect within one process; counter resets on container restart |
| Firestore counter with atomic `Increment(1)` | Free tier covers this scale (~$0/month) | New GCP dependency, IAM for the SA, one extra network hop per request | Correct across N instances; survives restarts |

`max-instances=1` makes the in-memory counter a single writer — no external
state needed. `concurrency=40` handles realistic bursts; Cloud Run queues
beyond that and returns 429 automatically. Scale-to-zero is preserved
because `min-instances=0`.

Counter persistence across container restarts is **intentionally none**. A
restart resets the counter. Worst case: two restarts in one day gives 3×
the daily cap of 200 requests = 600 requests × $0.002/query ceiling =
~$1.20 worst-case day, still comfortably inside the $10/month budget. This
is a documented tradeoff, not an accident.

**Migration path** (documented so future-me isn't surprised): if traffic
ever needs `max-instances > 1`, swap the in-memory counter for a Firestore
document with atomic `Increment(1)`. That is a ~30-line change plus a one-
time Firestore setup. It is deliberately deferred to keep the Phase 10
surface small.

## Cost guardrail numbers

**Per-query cost model** (from eval history and pricing in `pricing.py`):

- Non-corrective query: ~$0.0006 (~2K input tokens + ~500 output tokens on
  `gpt-4o-mini`)
- Corrective query (critic call + optional retry): ~$0.0015–0.002 worst-case
- Ceiling assumed for budgeting: **$0.002/query**

| Guardrail | Value | Rationale |
|---|---|---|
| Per-IP rate limit | `10/hour` sustained, `3/minute` burst | 10/hour is more than any real evaluator needs (a recruiter tries ~5 questions). 3/minute burst tolerates double-clicks and follow-up questions. Replaces the previous `60/minute` default which was sized for local development, not a public demo. |
| Global daily request cap | 200 requests/day across all IPs | At $0.002/query worst case → $0.40/day → $12/month. Sized just above amortized traffic for a portfolio demo. Counter resets at 00:00 UTC. 201st request returns HTTP 429 with a `demo_daily_limit_reached` error body. |
| Monthly budget ceiling | $10/month | Cloud Billing budget with email alerts at 50% ($5), 90% ($9), 100% ($10). At the daily cap this is a *soft* limit — the hard limit is the daily cap × 30 = $12 worst-case. Alerts give me time to react before overshoot. |
| Kill switch | `DEMO_ENABLED=false` env var | Flip in Cloud Run console → next request returns 503 `demo_disabled`. Zero-code emergency shutoff. |

## Cloud Run configuration (target values)

| Setting | Value | Reason |
|---|---|---|
| `min-instances` | 0 | Scale to zero when idle |
| `max-instances` | 1 | Single-writer property for the in-memory counter |
| `concurrency` | 40 | Handle realistic bursts; queue beyond that |
| `memory` | 1 GiB | Chroma + Python + Phoenix in one container fits comfortably |
| `cpu` | 1 | Baseline for the interactive demo; no need for allocated-during-idle CPU |
| `timeout` | 60 s | Longer than the LLM boundary timeout (`openai_timeout_seconds=20`) so we don't cut off a legitimate long generation |
| Secrets | `OPENAI_API_KEY` mapped from Secret Manager | Never bake secrets into the image |

## Post-approval implementation shape (informational)

The follow-up commits (approved separately) implement this ADR:

1. Tighten per-IP rate limit default in `config.py`.
2. Add `DailyBudget` and `DEMO_ENABLED` kill switch middlewares in
   `src/rag_harness/api/middleware/`.
3. Extend `QueryResponse` to include the per-stage trace, cost, and latency
   the demo UI needs.
4. Update `Dockerfile` to bake `chroma_db/` and drop `VOLUME` declarations.
5. Add `deploy/cloud-run.yaml` and a one-time GCP setup runbook.
6. Add the release-tag CD workflow that builds, gates on `make check` +
   eval, and deploys via Workload Identity Federation.

Custom domain (`rag.yashbhadiyadra.com`) is a final step after the app has
soaked at its `*.run.app` URL. That is not covered by this ADR.
