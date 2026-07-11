# ADR-0008 - Usage tracking via ContextVar collector

**Status:** Accepted  
**Date:** 2026-07-03  
**Decided by:** Owner, 2026-07-03

## Context

Phase 7 introduces per-call token and cost accounting. Every OpenAI API call
(in the generator, critic, corrective loop, HyDE, retriever query-embed, and
ingest embedder) needs to contribute a `TokenUsage` record that downstream
callers (the eval runner, the API metrics endpoint) can aggregate.

The obvious design is to change every function that touches the OpenAI SDK to
return `(result, TokenUsage)` or `(result, list[TokenUsage])`. That approach is
explicit and easy to reason about locally, but it ripples through every caller
that does not care about usage: the CLI query command, the FastAPI handler,
every test that mocks these functions, and every intermediate function that
composes them. On this codebase that is roughly 15 call sites and 80 tests.

The second option is a cross-cutting collector: an opt-in context manager that
sets a `ContextVar`, and an `record_usage(...)` helper that appends to the
current collector if one is active. This is the pattern OpenTelemetry uses for
span context propagation and is standard in production observability libraries.

## Decision

Use a `ContextVar`-based collector.

```python
with collect_usage() as usage_list:
    answer = generate(query, chunks)      # signature unchanged
    result = corrective_generate(...)      # signature unchanged
total_cost = sum(u.estimated_cost_usd for u in usage_list)
```

Instrumented call sites do exactly one thing after their OpenAI call:

```python
record_usage(TokenUsage.from_openai(model, response))
```

Callers that do not care about usage (CLI query, FastAPI handler, existing
tests) do not open a `collect_usage()` block and see no change in behavior.
Callers that do care (eval runner, `/metrics` endpoint) open a block, read
the list at the end, and aggregate.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Return `(result, TokenUsage)` from every LLM-calling function | Signature ripple across 15+ call sites and 80+ tests; every caller pays a syntactic cost for a feature it may not use |
| Return `(result, list[TokenUsage])` (list allows composition) | Same ripple; nested composition still leaks the concern up |
| Thread a `UsageCollector` object as an explicit parameter | Cleaner than tuple returns but same ripple; requires every intermediate function to forward it |
| `threading.local` collector | Works, but does not compose with async and does not reset on exception cleanly. `ContextVar` supersedes it in modern Python. |
| Global module-level list | No isolation: concurrent requests to the API server would contaminate each other |
| OpenTelemetry span attributes as the sole collector | Reasonable for tracing but requires the tracing backend to be up. We want usage tracking to work before the ADR-0009 tracing decision lands. |

## Consequences

- Call-site signatures do not change. The 80 existing unit tests do not need to
  be touched to compile; only the tests that specifically care about usage add
  a `collect_usage()` block.
- The collector is opt-in: outside a `collect_usage()` block, `record_usage()`
  is a no-op. This means "was this LLM call priced?" depends on whether an
  ancestor frame opened a collector, an implicit dependency that the reader
  has to be aware of. Documented in the module docstring and this ADR.
- Nested `collect_usage()` blocks produce nested independent collectors: an
  inner block does not see the outer collector and does not contaminate it on
  exit. This matches OpenTelemetry span-context semantics and is the intuitive
  behaviour for composition.
- Cost math lives in `observability/pricing.py`, decoupled from the call sites.
  Adding a new model or changing a price does not touch call sites.
- The FastAPI `/metrics` endpoint (Commit 5) opens a `collect_usage()` block
  around each `POST /query` handler, aggregates on completion, and updates
  Prometheus counters. Concurrent requests are isolated because ContextVar
  values are per-async-task (or per-thread) by construction.
