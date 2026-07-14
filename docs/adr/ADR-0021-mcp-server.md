# ADR-0021: MCP server

Date: 2026-07-14
Status: Accepted

## Context

Model Context Protocol became the de facto standard for exposing tools to
agents in 2026 (donated to the Linux Foundation, ~97M monthly SDK downloads,
supported across Claude, Cursor, VS Code, ChatGPT). Exposing the harness as
MCP tools makes it agent-consumable and is the distribution surface AI-engineer
tooling is expected to have. But the same year produced a grim MCP security
record: 30+ CVEs in two months, thousands of unauthenticated internet-exposed
servers, and real tool-poisoning and RCE incidents. So the server has to be
secure by construction, not just useful.

## Decision

Add an optional MCP server (`rag_harness mcp`, `[mcp]` extra) exposing a small
read-mostly tool set:

- `query_docs(question, top_k)` - answer from the pinned corpus with
  chunk-level citations.
- `get_eval_report()` - latest evaluation metrics and whether the gate passed.
- `get_ablation_report()` - the latest strategy-comparison table.

Secure by default:

- **stdio transport only.** The server communicates over stdin/stdout with
  the local MCP client that launches it; there is no network listener. Given
  the CVE record, refusing to open a port by default is the single most
  important control - there is nothing to expose.
- **Read-mostly surface.** No tool mutates state, runs a shell, writes files,
  or triggers an expensive job (a live `run_ablation` would be a multi-hour,
  cost-bearing operation and is deliberately not exposed; the read report is).
  There is no destructive action to authorise.
- **Bounded input.** `top_k` is clamped to the HTTP API's range.

`run_ablation` and other write/expensive tools, plus an HTTP transport, are
deliberately excluded. Before any HTTP exposure the CSA baseline applies:
OAuth 2.1 + PKCE, tool-level (not server-level) scopes, access-token lifetime
<= 1 hour, and default-deny filesystem and network for the server runtime.
That is documented here as the gate, not implemented, because the secure
default (stdio, read-only) needs none of it.

## Consequences

- The harness is now agent-consumable: an agent can retrieve grounded,
  cited answers and read the reliability numbers through a standard protocol.
- Optional dependency: the `mcp` SDK is a `[mcp]` extra, lazy-imported, so the
  base install and CI stay lean. The tool implementations are tested without
  the SDK; the server construction test skips where the extra is absent.
- The security posture is the differentiator: "the MCP server that refuses to
  open a port and exposes nothing destructive" is a credible stance against
  the 2026 MCP threat landscape, and it ties directly to the Phase 2 security
  work.
