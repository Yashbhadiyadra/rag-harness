# ADR-0018: OpenTelemetry GenAI semantic conventions

Date: 2026-07-14
Status: Accepted

## Context

The tracing layer (ADR-0009) already emits OpenTelemetry spans and, when
enabled, exports them to Phoenix over OTLP. But the LLM-call attributes on
those spans used ad hoc names, so a trace was only meaningfully queryable
in the Phoenix backend this project happens to configure. OpenTelemetry
has since standardised GenAI semantic conventions
(`gen_ai.*`), and industry observability platforms (Microsoft Foundry,
Langfuse, Arize) have converged on them - emitting OTEL-compatible GenAI
traces is now table stakes for eval/observability tooling.

## Decision

Annotate LLM calls with the OpenTelemetry GenAI semantic-convention
attributes on the currently active span:

- `gen_ai.system` = `openai`
- `gen_ai.operation.name` = `chat` (or `embeddings`)
- `gen_ai.request.model`
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`

A new `observability/semconv.py` holds the attribute-key constants and two
helpers: `genai_attributes(...)` builds the dict, and
`set_current_genai_attributes(...)` sets them on `trace.get_current_span()`.
The generation boundary (`generate_async`) calls the setter after each
completion with the real token counts. Setting on the *current* span is the
idiomatic OTEL pattern: the LLM call annotates whatever stage span
(`generate`, `retrieve`, ...) is active, and outside a recording span the
current span is a non-recording no-op, so the call is unconditional and
free when tracing is off.

## Consequences

- Traces exported to any OTEL backend now carry the standard `gen_ai.*`
  attributes, so model, operation, and per-call token usage are queryable
  by the same names everywhere, not just in Phoenix. This is the
  portability the strategy research flagged as table stakes.
- No new dependency: the OpenTelemetry API is already a core dep.
- The API-response trace (the `TraceSpan` list the demo UI renders) is
  unchanged - it records the stage kwargs, not the OTEL span's attributes,
  so the demo UI is unaffected. The GenAI attributes are a backend-export
  concern.
- The setter is wired at the generation boundary now. Extending it to the
  HyDE, decomposition, judge, and embedding calls is a mechanical follow-up
  using the same helper; the pattern is established and tested.
