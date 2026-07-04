# Production Roadmap — RAG Reliability Harness v1.0

> Status: Phases 1–6 complete (ingest, five retrieval strategies, grounded + corrective
> generation, three-metric eval core, API/CLI, cache, logging, Docker). This document
> defines the path from "working repo" to **deployed, measured, showcasable product**.
>
> Governing rule (non-negotiable): **no new capability lands until the previous one is
> measured.** Features without numbers are liabilities in this project, not assets.

---

## Product definition

**One-liner:** A production RAG system over the Kubernetes documentation that *measures
its own reliability* — decomposed failure-mode metrics, regression gates in CI, live
observability, and published evidence for every architectural claim.

**Who it's for (product framing):** any team shipping RAG who needs to answer "did this
change make retrieval better or worse?" with a number. The K8s corpus is the demo domain;
the harness pattern is the product.

**The 90-second demo story (what a visitor experiences):**
1. Open the live URL → ask a real K8s question → watch the answer arrive with its
   retrieved sources and a per-stage trace (retrieve → critique → generate).
2. Open the public metrics page → see faithfulness / recall / correctness across
   strategies, cost-per-query, p50/p95 latency, and the eval-score trend over time.
3. Open the repo → CI badges green, eval-gate workflow visible, 7+ ADRs, ablation table
   in the README proving which retrieval strategy wins and what it costs.

That trio — *touch it, see the evidence, read the engineering* — is the showcase.

---

## Phase 7 — Measure everything (observability core)

The current gap: quality metrics exist, operational metrics don't. A production system
knows its latency, tokens, and cost per query.

- Extend `EvalResult` and the query path with `latency_ms`, `input_tokens`,
  `output_tokens`, `estimated_cost_usd` (priced from a model-rates table in config).
- Add `answer_relevancy` and `context_precision` to complete the metric suite
  (retrieval: recall + precision · grounding: faithfulness · generation: correctness
  + relevancy).
- Per-stage tracing: spans for retrieve / critique / generate with timings, wired to a
  self-hosted trace viewer (Phoenix or Langfuse — self-hosted keeps cost ~zero; record
  the choice as an ADR).
- `GET /metrics` endpoint (Prometheus text format) + request counters and error rates.
- Eval summary reports p50/p95 latency and mean cost alongside quality metrics.

**Exit criterion:** a single eval run prints quality + latency + cost in one table.

## Phase 8 — Prove it (evaluation completeness)

The current gap: five strategies and a corrective loop exist; none has published evidence.

- Plumb `corrective_generate` into `run_eval` (same `--corrective` flag); report paired
  per-case deltas vs. baseline.
- **Ablation runner:** one command runs the golden eval across `dense`, `hybrid`,
  `hybrid-rerank`, `hyde`, `full` × {baseline, corrective} and emits a comparative
  table (markdown + CSV): quality metrics, cost, latency per configuration.
- Persist eval history (append-only JSONL in `evals/history/`), so trends are plottable
  and regressions attributable to commits.
- Wire the eval gate to enforce thresholds on a cheap subset per-PR (mocked/LLM-light),
  full suite nightly — thresholds live in config, failure blocks merge.
- Golden set: owner hand-reviews all cases (eval discipline requirement); grow toward
  50+ cases including adversarial ones (ambiguous, multi-hop, version-sensitive,
  unanswerable — the unanswerable cases prove the refusal path works).
- **Measure `answer_relevancy` × `correctness` divergence explicitly.** Treat
  "relevant but incorrect" — high relevancy, low correctness — as a highlighted
  failure category in the ablation output. That combination is confident-sounding
  hallucination: the answer addresses the question but gets the facts wrong,
  which is a distinct and more dangerous failure than an off-topic response.
  Per-case flags in the CSV + a dedicated row in the markdown table.

**Exit criterion:** the README ablation table exists with real numbers, a PR that
degrades faithfulness demonstrably fails CI, and the "relevant but incorrect"
failure count is reported per configuration.

## Phase 9 — Harden it (production resilience)

The current gap: happy-path service. Production means "what happens when things break."

- Async request path end-to-end (async OpenAI client, async endpoint handlers);
  bounded concurrency for batch operations.
- Resilience at the LLM boundary: retries with exponential backoff + jitter, timeouts,
  circuit-breaker-style fallback; on retrieval failure or low-confidence critique,
  degrade to the honest refusal path (already built — make it the documented behavior).
- Rate limiting on the public API; request size/length caps.
- Input guardrails (minimal, deliberate): query length limits + prompt-injection
  screening at the boundary; document scope explicitly (full guardrails engine is a
  separate future project).
- Error taxonomy: typed exceptions → correct HTTP codes → structured log events.
- `GET /health` (liveness) vs `GET /ready` (dependencies reachable) split.
- Load sanity check: a short script proving the service holds N concurrent requests;
  record results.

**Exit criterion:** kill the vector store or the LLM key mid-demo and the service
answers gracefully instead of 500ing.

## Phase 10 — Ship it (deployment)

- Multi-stage Dockerfile (slim runtime image); image build in CI.
- Deploy API to a low-cost managed runtime (Cloud Run or Fly.io — decision ADR; Cloud
  Run aligns with existing GCP familiarity and scales to zero).
- Chroma persistence strategy for deploy (baked snapshot volume vs. rebuild-on-boot vs.
  hosted store) — ADR with cost analysis.
- CI/CD: GitHub Actions pipeline — on release tag: build → test → eval gate → deploy.
- Secrets via platform secret manager; hard **budget caps** and spend alerts on the
  LLM key; per-IP rate limits so a public demo can't drain the budget.
- **Demo UI:** minimal, clean web UI (single-page; server-rendered or small React app)
  showing answer + sources + per-stage trace + cost/latency of *your* query. Judged as
  a product surface, so design matters more than features.
- **Public metrics page:** static dashboard generated from eval history (charts of
  metric trends, the ablation table, cost/latency) — regenerated by CI nightly.

**Exit criterion:** a stranger with the URL can ask a question, see the trace, and
browse the evidence — with zero setup.

## Phase 11 — Showcase it (product layer)

- README rewritten as an ops-grade document: architecture diagram, live links, metrics
  screenshots, "how the eval gate works," runbook notes, ablation table front and
  center.
- `v1.0.0` tagged release with real release notes; changelog current.
- Demo script: the 90-second walkthrough, written down (doubles as interview narrative).
- Content: ablation study published as a technical post; corrective-RAG delta as a
  second post; both sourced from the repo's own numbers.

**Exit criterion:** someone who has never met the author understands, within two
minutes, what this is, why it's hard, and that it's real.

## Phase 12 — Extend (bridges, strictly after v1.0)

- **Stale-embedding hooks → Project 2:** the provenance fields + embedding cache are
  the input signal; begin the standalone library.
- **Agent-eval reuse → Project 3:** evaluation layer generalizes to trajectory scoring.
- **MCP server (stretch, high-signal):** expose the harness as MCP tools
  (`query_docs`, `get_eval_report`, `run_ablation`) so agents can consume it — connects
  the project to the agent-interop standard and demonstrates MCP skill hands-on.

---

## Skill-stack coverage map (role checklist → where this project proves it)

| Skill screened for | Proven by |
|---|---|
| RAG architecture, hybrid retrieval, reranking | Phases 1–6 + ablation evidence (P8) |
| Eval design (top differentiator) | Metric suite, golden set, CI eval gate, ablation (P8) |
| Production observability | Tracing, /metrics, latency/cost instrumentation (P7) |
| Cost optimization | Cost-per-query metric, caching, model-rates config, budget caps (P7/P10) |
| Safety/guardrails (baseline) | Input guardrails, refusal path, rate limits (P9) |
| Async + production Python | Async path, resilience patterns, typed errors (P9) |
| CI/CD + deployment | Actions pipeline, Docker, Cloud Run, release discipline (P10) |
| Agentic patterns | Corrective critic-retry loop (P6) + MCP stretch (P12) |
| Communication / product sense | Demo UI, metrics page, ops README, demo story (P10–11) |

## Senior signals (how the work is done — matters as much as what)

- Small PRs, Conventional Commits, green CI before merge — no exceptions.
- Every architectural choice has an ADR; every capability claim has a number.
- Bugs get a failing test before a fix.
- Backlog of the three documented ADR gaps (BM25 tokenizer, factory pattern,
  integration-test policy) closed opportunistically.
- Makefile venv defect fixed immediately (`uv run` prefixes restored); stale remote
  branches pruned.

## Sequencing & sizing

P7 → P8 are the immediate priority (measurement before anything new — they convert
existing features into evidence). P9 → P10 make it survive and ship. P11 packages it.
At the current build pace: roughly 4–6 focused weeks to v1.0, keeping Projects 2 and 3
on schedule. Scope beyond this document goes to the backlog, not the sprint.
