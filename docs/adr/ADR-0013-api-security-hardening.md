# ADR-0013: API security hardening baseline

Date: 2026-07-13
Status: Accepted

## Context

A security audit of the API surface ahead of v1.0 confirmed the
existing controls (per-IP rate limiting, daily request cap, kill
switch, input length bounds, injection screening, typed errors without
stack traces, pinned corpus with SHA verification, pip-audit in CI) and
found three gaps: no security response headers, two redundant
API-documentation surfaces, and no written statement of the service's
security posture.

## Decision

1. **Security headers on every response** via a dedicated outermost
   middleware: `X-Content-Type-Options: nosniff`,
   `X-Frame-Options: DENY`, one-year HSTS, `Referrer-Policy:
   no-referrer`, and a strict same-origin Content-Security-Policy
   (`frame-ancestors 'none'`, no inline-script or inline-style
   carve-outs - the demo UI uses only `/static` assets). Values are
   constants, not settings: no deployment of this service wants weaker
   headers, so configurability would only invite drift.

2. **`/docs` stays enabled; `/redoc` is removed.** The repository is
   public, so the OpenAPI schema reveals nothing that is not already
   in source, and interactive docs are part of the showcase. Two
   documentation UIs is one more attack-and-maintenance surface than
   necessary.

3. **Security posture documented in the README** (see "Security"
   section): what data leaves the service (queries and retrieved
   chunks to the LLM API - nothing else), what is logged (60-char
   query prefixes at most on warning paths), what `/metrics` exposes
   (aggregate counters only), and the secure-by-default stance that
   insecure configurations require explicit opt-out.

## Consequences

- Every response, including middleware short-circuits (rate-limit,
  kill-switch, daily-cap rejections), carries the header set;
  regression-tested in `tests/test_security_headers.py`.
- The CSP forbids inline scripts and styles: any future demo-UI change
  must keep JS/CSS in `/static` files or consciously amend the policy
  here.
- Prompt-hardening of retrieved chunks ("passages are data, not
  instructions") is deliberately deferred: prompt changes shift eval
  scores, so it lands together with the poisoned-corpus evaluation
  that can measure it (post-v1.0 roadmap), not as an unmeasured tweak.
