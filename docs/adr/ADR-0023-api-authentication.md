# ADR-0023: API authentication with hashed API keys

Date: 2026-07-15
Status: Accepted

## Context

Until now `/query` has been protected only by per-IP rate limiting, a
daily request cap, and a kill switch (ADR-0010, ADR-0013). That is the
correct posture for a deliberately public, scale-to-zero demo, but it is
the first blocker on the path from demo to product: there is no notion of
a caller identity, so quotas cannot be fair across clients behind a shared
NAT, access cannot be revoked per client, and usage cannot be attributed.

This ADR adds authentication as an opt-in capability. It must not break the
existing public demo (which is a showcase feature), and it must be the
minimal secure step - not a user/identity system, which multi-tenancy
(future work) will need but this phase does not.

## Decision

1. **Header-based API keys.** Callers present a key as
   `Authorization: Bearer <key>`. Chosen over a bespoke `X-API-Key` header
   because the standard `Authorization` header is understood by proxies,
   client SDKs, and log-redaction tooling, and it is what comparable
   key-based APIs (OpenAI) use. Chosen over JWT/OAuth2 because there is no
   identity provider or login flow, and API keys are the right primitive
   for programmatic server-to-server access. The scheme is swappable for
   tokens later without changing call sites (the dependency is the seam).

2. **Keys are stored only as hashes.** Configuration holds a set of
   SHA-256 hex digests (`API_KEYS`, comma-separated), never plaintext keys.
   An incoming key is hashed and checked for set membership. There is no
   database: this is a hashed allowlist in config, provided via env/secret.
   A documented one-liner (and a small `hash-key` CLI helper) produces a
   digest from a fresh key so operators never paste plaintext into config.
   Multi-tenancy (future work) will replace the allowlist with a
   DB-backed store that supports per-tenant issuance and rotation; the
   verification seam stays the same.

3. **Enforcement is explicit and fails safe when misconfigured.** A new
   `API_AUTH_ENABLED` setting (default `false`) preserves the current open
   demo. When `true`, `/query` requires a valid key or returns `401` with a
   `WWW-Authenticate: Bearer` header. Enabling auth with an empty allowlist
   is a configuration error: the app raises at startup rather than run
   "authenticated" with no keys - you cannot believe you are protected when
   you are not. When auth is disabled, startup logs a one-time warning so an
   operator never leaves it off by accident.

4. **Only `/query` is authenticated.** `/health`, `/ready`, and `/metrics`
   stay open: liveness/readiness probes and Prometheus scrapers cannot carry
   a key, and `/metrics` exposes only aggregate counters (ADR-0013). The
   demo UI at `/` stays open; when auth is on it is a thin client that must
   supply a key.

5. **Rate limiting keys on the caller identity.** The slowapi `key_func`
   returns a per-key bucket for authenticated requests and falls back to the
   client IP for anonymous ones. To avoid coupling to dependency-resolution
   order, the `key_func` parses and validates the header itself rather than
   reading state the endpoint dependency sets. Authenticated clients thus
   get their own fair quota; anonymous traffic is still bounded by IP.

## Consequences

- Keys never touch the logs or `/metrics`, and never exist as plaintext at
  rest. Verification uses `hmac.compare_digest` on the hashed value to keep
  comparison constant-time.
- Default `API_AUTH_ENABLED=false` means this capability ships off and the
  public demo is unchanged. Production deployments set it `true` and provide
  `API_KEYS`; `deploy/` docs and `.env.example` will state this as required
  for any non-demo deployment.
- The measurement (governing rule) is an auth test matrix: valid key -> 200,
  missing key -> 401, malformed/unknown key -> 401, per-key quota isolation
  (two keys do not share a bucket), probes stay reachable without a key, and
  enabling auth with no keys fails at startup. Because auth does not touch
  retrieval or generation, the reliability eval gate must show no metric
  movement - confirming auth is orthogonal to answer quality.
- MCP server (ADR-0021) is unaffected: it is stdio-only, local, and exposes
  no network listener, so there is no unauthenticated network surface there.
- This is not multi-tenancy: every valid key sees the same corpus. Per-tenant
  isolation is deliberately deferred to the phase that introduces a real key
  store.

## Alternatives considered

- **JWT/OAuth2 now:** rejected as premature - no IdP, no login, heavier
  operationally, and the wrong primitive for programmatic access.
- **DB-backed key issuance now:** rejected for this phase - there is no
  persistence layer for it yet, and a hashed allowlist delivers the security
  property (revocable, no plaintext) at a fraction of the cost. Folded into
  multi-tenancy.
- **Fail-closed by default (auth always on):** rejected because it would
  break the intentional public demo. Explicit opt-in with a
  fail-at-startup-on-misconfiguration guard gives the same safety without
  regressing the showcase.
