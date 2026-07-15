# ADR-0026: Secrets management posture

Date: 2026-07-15
Status: Accepted

## Context

The service needs an `OPENAI_API_KEY` at runtime, and a scaled-out deployment
(ADR-0024) may need a `REDIS_URL` that carries credentials. A product must not
bake secrets into the image or expose them as plaintext at rest.

Much of this is already true and was established with the deployment work
(ADR-0010), but it had never been recorded as a decision, one security note in
the README had gone stale, and `REDIS_URL` post-dates that work. This ADR
records the posture and closes those gaps.

## Decision

1. **Secrets are resolved by the platform, not fetched by the application.**
   On Cloud Run, `OPENAI_API_KEY` is injected from Secret Manager via the
   manifest's `valueFrom.secretKeyRef` at cold start (`deploy/cloud-run.yaml`),
   and the runtime service account is a scoped `secretmanager.secretAccessor`
   for exactly that secret (`deploy/README.md`). The application simply reads
   the key from its environment. We deliberately do NOT add an in-app
   secrets-manager client: it would duplicate what the platform does, add a
   dependency and IAM surface, and is unnecessary when the runtime injects the
   value. Rotation is "add a new secret version"; the container reads `latest`
   at the next cold start.

2. **The image never contains a secret.** `.env` is in `.dockerignore`, the
   Dockerfile bakes no key, and `.env` is the local-development fallback only.
   A guard test asserts these properties so a regression (removing `.env` from
   `.dockerignore`, or hardcoding a key) fails CI.

3. **`REDIS_URL` is treated as a secret when it carries credentials.** For a
   scaled-out deployment (ADR-0024), an authenticated Redis URL
   (`redis://user:pass@host`) is injected from Secret Manager the same way as
   the OpenAI key, not set as a plaintext env value. The manifest documents the
   pattern; it is commented out because Redis is opt-in and not part of the
   default single-instance demo.

4. **`API_KEYS` / `TENANTS` are not secrets.** They contain only SHA-256
   digests (ADR-0023, ADR-0025), never plaintext keys, so they are
   configuration rather than secrets and do not require Secret Manager. The raw
   keys they authenticate are held by clients and by whoever provisions them,
   never by the service.

## Consequences

- The production key lives only in Secret Manager; it is not in the image, the
  manifest, the repo, or the container filesystem. `/metrics` and logs already
  never emit it (ADR-0013).
- The measurement (governing rule): a guard test (`tests/test_image_secrets.py`)
  asserting `.env` is git- and docker-ignored and the Dockerfile hardcodes no
  key, plus the existing deploy runbook that provisions the secret and the
  scoped reader role. This makes "no secret in the image" a checked property,
  not a hope.
- The README security section now states the real posture (Secret Manager in
  production, `.env` for local dev) instead of the stale "keys live only in
  `.env`".
- Local development and tests are unchanged: they read `OPENAI_API_KEY` from
  `.env` or the environment, with no cloud dependency.

## Alternatives considered

- **An in-app secrets-manager client** (fetch the key from the Secret Manager
  API at boot). Rejected: it duplicates the platform's native injection, adds a
  `google-cloud-secret-manager` dependency and code path, and widens the IAM
  surface, for no benefit on Cloud Run. It would only matter on a platform
  without native secret injection, which is not where this ships.
- **Storing the key as a plain Cloud Run env value.** Rejected outright: that
  is the plaintext-at-rest exposure this posture exists to prevent.
