# ADR-0009 — Tracing backend: Arize Phoenix

**Status:** Accepted  
**Date:** 2026-07-03  
**Decided by:** Owner, 2026-07-03

## Decision

**Arize Phoenix** was chosen over Langfuse. Rationale: operational fit with
the "self-hosted keeps cost ~zero" rule, cleaner Cloud Run deploy story
(SQLite default vs Postgres required), and OpenTelemetry-native
instrumentation preserves reversibility if we later want to swap backends.

The comparison analysis below is preserved for future reference.

## Context

Phase 7 delivered per-call token accounting, per-query latency, and cost
metrics. What's still missing is **per-stage tracing**: a span tree that shows,
for a single query, how long retrieval took, how long the critic took, how
long generation took, what tokens flowed where. That's the artefact a demo
visitor should be able to see on the public metrics page (per ROADMAP.md
Phase 10) and the artefact an interviewer will ask about when reviewing this
project.

The industry-standard shape is OpenTelemetry spans: a span per stage, nested
by parent-child relationship, exported to a UI that renders the tree with
timings. Two self-hosted, LLM-aware backends are appropriate for this
project's constraints ("self-hosted keeps cost ~zero"). This ADR picks
between them.

Both backends accept OpenTelemetry spans; the choice does not lock in the
instrumentation format. That is a good property: it means the wrong choice
here is recoverable.

## Options

### Option A — Arize Phoenix

- **License:** Apache 2.0
- **Install:** `pip install arize-phoenix`. Pure-Python; launches a local UI
  from the same process. No separate service required for a demo.
- **Storage:** SQLite by default (single file); can swap to Postgres for
  multi-instance deploys.
- **Docker footprint:** ~200 MB image for the standalone server mode.
- **Instrumentation:** OpenTelemetry-native via `openinference-instrumentation-openai`
  and `openinference-instrumentation-llamaindex` etc. Framework-specific
  instrumentors are auto-instrumenting for the common libraries.
- **UI:** span waterfall, prompt/completion side-by-side view, per-span
  latency and token counts, trace search by attribute.
- **Portfolio-visibility:** less well-known than Langfuse but growing;
  Arize the company sells enterprise observability so Phoenix has healthy
  investment.
- **Deploy story for Phase 10:** simple. A single container plus a shared
  volume. Scales-to-zero on Cloud Run cleanly because SQLite is fine for a
  demo.
- **Local-dev friction:** one command (`phoenix serve`) or one line of code
  (`px.launch_app()`). Very low.

### Option B — Langfuse

- **License:** Apache 2.0 (core; some enterprise features are commercial)
- **Install:** SDK is `pip install langfuse`. Server is a separate Docker
  Compose stack requiring Postgres.
- **Storage:** Postgres (mandatory).
- **Docker footprint:** ~600 MB total including Postgres.
- **Instrumentation:** SDK-first; wrap decorators around functions. Native
  OpenTelemetry support was added in 2024 but the idiomatic path is still
  the SDK.
- **UI:** span waterfall, prompt versioning, dataset management, LLM-based
  evaluations built into the UI, session grouping, user tracking.
- **Portfolio-visibility:** stronger. More common in job postings, more
  frequently referenced in engineering blogs, larger community.
- **Deploy story for Phase 10:** heavier. A Compose stack with Postgres is
  awkward on Cloud Run. Fly.io handles Postgres more naturally.
  Alternatively, use Langfuse Cloud's free tier and skip self-hosting.
- **Local-dev friction:** medium. Must launch Postgres and the Langfuse
  server before the app can emit traces. `docker compose up` is one command
  but it's not the zero-config launch that Phoenix offers.

## Decision dimensions

| Dimension | Phoenix | Langfuse | Winner |
|---|---|---|---|
| Cost-to-run (dev + demo) | ~0 (SQLite) | Postgres required | **Phoenix** |
| Cost-to-integrate | 1 line of code | SDK wrap or OTel setup | **Phoenix (slight)** |
| UI quality for the 90-second demo | Waterfall + prompt view | Waterfall + prompt view + evals + versioning | **Langfuse** |
| Portfolio-signal weight (visibility in job postings) | Lower | Higher | **Langfuse** |
| Fit with the "self-hosted keeps cost ~zero" rule | Fits cleanly | Works but requires Postgres | **Phoenix** |
| Deploy story on Cloud Run (Phase 10) | Simple single container | Postgres awkward on Cloud Run | **Phoenix** |
| Deploy story on Fly.io (Phase 10 alternative) | Simple | Native Postgres → clean | **Tie** |
| Docker footprint | ~200 MB | ~600 MB | **Phoenix** |
| Ecosystem breadth (evals, prompt versioning) | Growing | Broader today | **Langfuse** |
| Long-term reversibility if we switch later | High (OTel native) | Medium (SDK migration) | **Phoenix (slight)** |

Phoenix wins on operational fit and cost. Langfuse wins on portfolio-signal
weight and feature depth.

## Rejected alternatives

| Option | Why rejected |
|---|---|
| **LangSmith** | Commercial, monthly cost. Roadmap explicitly rules out paid backends for this project. |
| **OpenTelemetry + Jaeger** | Generic tracing; UI is not LLM-aware. No prompt/completion view, no token/cost attribution built in. Reinventing the LLM-specific dashboard is out of scope. |
| **W&B Weave** | Requires W&B account; free tier exists but couples the demo to a third-party dashboard we don't control. Undesirable for a self-hosted showcase. |
| **Roll our own**: extend the existing observability layer with JSON-lines file dumps and a hand-rolled HTML viewer | Reinventing a wheel that is already free. Not defensible in an interview. |

## My recommendation (marked for review, not chosen for you)

**Phoenix.** Reasons in priority order:

1. **Operational fit with the "self-hosted keeps cost ~zero" rule.** SQLite
   default vs Postgres-required is the biggest divider. For a demo instance
   that scales to zero, Phoenix's single-container model is measurably
   simpler.
2. **Deploy story on Cloud Run.** Roadmap Phase 10 names Cloud Run as the
   preferred runtime. Cloud Run + Postgres is a non-trivial architectural
   choice (Cloud SQL costs $8+/month minimum, or you run Postgres in the
   container and lose scale-to-zero). Phoenix + SQLite + a persistent volume
   is a single moving part.
3. **OpenTelemetry-native.** If we later decide Langfuse's feature depth
   matters, we can point the same OTel exporter at Langfuse without
   rewriting the instrumentation. Reversibility is a real property here.
4. **Feature-depth gap is closable.** Langfuse's prompt versioning and
   dataset management are attractive but not required for the Phase 7 exit
   criterion, the Phase 8 ablation, or the Phase 10 demo. We are not building
   a prompt-ops product.

**The case for overriding my recommendation:** if the goal is maximum
portfolio signal to hiring managers rather than lowest operational cost,
Langfuse's higher industry visibility is worth the deploy overhead. The
Docker Compose file for Langfuse is not complicated; Postgres on Fly.io is
free-tier-friendly. If we're planning to deploy on Fly.io anyway, this
tradeoff softens considerably.

Choose Phoenix if you want the fastest path to a demoable trace tree with
minimal ops surface. Choose Langfuse if the signal value of "we use
Langfuse" outweighs one extra moving part.

## Post-approval implementation shape (informational, not scoped)

Once you pick, the follow-up work is:

- Add either `arize-phoenix` or `langfuse` to core deps.
- New `src/rag_harness/observability/tracing.py`: a `tracer` singleton and a
  `@traced(stage: str)` decorator that opens a span, attaches
  `strategy`/`top_k`/`corrective` attributes, and records latency and token
  counts on span end.
- Wrap retrieve / critique / generate stages with the decorator.
- Add a docker compose file (or extend the existing one) for the tracing
  backend so `docker compose up` gives a full observability stack.
- Change this ADR's status from Proposed → Accepted with the chosen option
  called out.
- Ship as a single commit tagged `feat(observability): add per-stage tracing`.

Estimated size: ~200 LOC + 8 tests. One commit, one dependency, ~half a day
of focused work.

