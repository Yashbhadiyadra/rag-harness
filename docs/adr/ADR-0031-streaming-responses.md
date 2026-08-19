# ADR-0031: Streaming answers over a separate SSE endpoint

Date: 2026-08-19
Status: Accepted

## Context

The system is now deployed to a public URL and the live demo is the top of the
funnel: a visitor forms an opinion in the first few seconds. The measured p50
end-to-end latency of `/query` on the dense strategy is ~4.4s (160-case golden
sweep, commit 164ff8c), almost all of it in generation. Today the demo UI does a
blocking `POST /query` and shows a spinner for the whole 4.4s before any text
appears. Time-to-first-token, not total latency, is what a visitor feels.

`/query` returns a rich JSON body (answer + sources + citations + trace +
cost_usd + latency_ms) that eval, the test suite, the MCP server, and API clients
all depend on. Changing that contract to stream would risk the measured
reliability path for a pure-UX gain.

## Decision

Add a sibling endpoint `POST /query/stream` that returns Server-Sent Events, and
leave `/query` unchanged. The reliability path, eval, and API contract are
untouched; streaming is additive and demo-facing.

SSE event protocol (`text/event-stream`):

- `sources` - emitted immediately after retrieval, before generation, so the UI
  can paint provenance in ~1s instead of after the full answer.
- `token` (repeated) - answer text deltas as the model produces them.
- `done` - final metadata: `citations` (parsed from the assembled answer),
  `trace`, `cost_usd`, `latency_ms`, `ttft_ms`.
- `error` - mirrors the existing `error_type` contract (daily-cap, guardrail,
  429, LLM-refusal) so the client's error handling is unchanged.

Generation adds `generate_stream(query, chunks) -> AsyncIterator[str]` using the
OpenAI streaming API with `stream_options={"include_usage": True}`, so per-query
usage and cost are still recorded from the final chunk. `generate_async` is kept
unchanged for eval, tests, and the non-stream path.

The stream endpoint serves the **non-corrective** path only. The corrective loop
is multi-step (generate -> critique -> regenerate), does not stream cleanly, is
off by default, and is measured to *lower* correctness on this corpus (0.917 ->
0.906); `/query` continues to serve it. It reuses the same injection screen,
retriever, tenant dependency, rate limit, daily-cap/kill-switch middleware, and
`collect_spans()` / `collect_usage()` collectors as `/query`.

`X-Accel-Buffering: no` is set so Cloud Run does not buffer the stream.

## Consequences

Streaming changes *perceived* latency, not reliability: no faithfulness,
correctness, recall, or citation-accuracy number moves, and the eval gate is
unaffected by design. The measurement for this change is a new Prometheus
histogram `rag_query_stream_ttft_seconds` (time to first token), reported before
and after against the local container and the live service.

Measured on the same question ("How do I roll back a Deployment in Kubernetes?"),
warm instance:

| Path | What the visitor waits for | Local | Live (Cloud Run) |
|---|---|---|---|
| `/query` (blocking) | full answer, nothing before it | 3415 ms | 7475 ms |
| `/query/stream` | sources rendered | 258 ms | 544 ms |
| `/query/stream` | first answer token | 1017 ms | 1109 ms |

Time-to-first-token is ~1.0-1.1s versus 3.4-7.5s for the blocking full answer, and
sources paint in ~0.3-0.5s where the blocking path showed nothing until the whole
answer was ready. No reliability number moved. The blocking total varies with
answer length and API latency, so the honest headline is TTFT and time-to-sources,
not a total-latency comparison.

Cold start from scale-to-zero is unaffected - streaming hides the generation wait,
not the container warm-up; that remains a separate concern (min-instances would
trade cost for cold-start and is out of scope here).

The cost of the addition is a second answer code path to maintain. It is bounded:
`generate_stream` shares `_build_context` and the system prompt with
`generate_async`, and the endpoint shares every dependency with `/query`.
