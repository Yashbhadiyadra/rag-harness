# ADR-0028: Stateless HTTP MCP server

Date: 2026-08-07
Status: Accepted

## Context

The MCP server (ADR-0021) is stdio-only by design: no network listener, the
strongest control against the 2026 record of unauthenticated internet-exposed
servers. But the product thesis is a hosted, trustworthy retrieval MCP endpoint
that an external agent connects to over the network, so a network transport has
to exist eventually.

Two things make now the right time to build it against the right shape. First,
the MCP spec release candidate 2026-07-28 rewrote the protocol core to be
**stateless**: a remote server can run behind a plain round-robin load balancer
with no sticky sessions or shared session store, and it hardened authorization
around OAuth 2.1 (RFC 9207 issuer validation, credential-to-auth-server
binding). Building the HTTP surface stateless-native now avoids shipping the
sticky-session design the spec just deprecated. Second, the codebase already has
the pieces a trustworthy multi-tenant endpoint needs: a hashed key allowlist
(ADR-0023) and per-tenant corpus isolation (ADR-0025).

Note the spec is a release *candidate*. The transport shape (stateless,
header-routed) is stable enough to build on; the full OAuth 2.1 resource-server
flow is not something we implement here, only leave room for.

## Decision

Add a **stateless** streamable-HTTP transport for the same tool set, alongside
the stdio server, not replacing it. `FastMCP(stateless_http=True)` provides the
stateless ASGI app; there is no session store, so each request is
self-contained and the service scales horizontally without shared state.

Authentication reuses the existing primitives rather than inventing new ones. A
Starlette middleware extracts the `Authorization: Bearer <key>` token and
resolves it to a `TenantContext` via `resolve_tenant` (ADR-0023/0025). When
`API_AUTH_ENABLED` is on, a missing or unknown key is rejected with 401 and
`WWW-Authenticate: Bearer` before any tool runs; when off (the public demo) the
request resolves to the default tenant. The resolved tenant is placed in a
`ContextVar` for the duration of the request, and the query tool reads it to
select that tenant's Chroma collection, so one agent can never retrieve another
tenant's corpus. The full OAuth 2.1 resource-server flow the spec hardens is a
documented follow-up; bearer-over-TLS reusing the hashed allowlist is the
minimum that is genuinely safe to expose and testable today.

Tenant is resolved server-side from the credential, never a client-supplied
tool argument, so the exposed tool schema is identical for every caller and a
client cannot request a collection it was not issued.

Trust is surfaced in the payload: `query_docs` returns, alongside the answer and
citations, the **provenance** of the chunks it used (the corpus git commit and
doc version). An agent, or the human behind it, can verify which pinned corpus
snapshot produced the answer. This is the "payload trust, not just pipe trust"
differentiator expressed in the protocol response.

Tools stay read-mostly and input stays bounded (`top_k` clamped), exactly as the
stdio server. The HTTP app is exposed via `rag-harness mcp-http`; wiring it to
Cloud Run and listing it in the MCP registries are separate steps gated on
deploy and demand.

## Consequences

- An external agent can connect over streamable HTTP, authenticate with a
  bearer key, and retrieve grounded, cited answers scoped to its tenant, with
  the corpus provenance in the response. Verified locally against the running
  ASGI app (auth rejects missing/invalid keys; a valid key resolves its tenant;
  cross-tenant retrieval is impossible because the collection is derived from
  the credential, not the request body).
- Stateless by construction: no session store, so the service can run behind a
  round-robin load balancer and scale to more than one instance without shared
  state, matching the 2026-07-28 transport model and unblocking a future
  `maxScale > 1` (ADR-0024).
- The stdio server is unchanged and remains the default for local clients.
- Not done here, and deliberately: full OAuth 2.1 authorization-server
  integration, Cloud Run exposure, and registry listings. Each is gated on the
  deploy landing or a customer pulling for the hosted endpoint, per the founding
  plan's guardrail against building product surface ahead of demand.
