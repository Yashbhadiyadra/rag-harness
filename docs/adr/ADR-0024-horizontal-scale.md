# ADR-0024: Horizontal scale via shared rate-limit and budget state

Date: 2026-07-15
Status: Accepted

## Context

Two pieces of request-governing state are held in the process today:

- The slowapi rate limiter uses in-memory storage (per-instance counters).
- The daily request cap is an in-process `DailyBudget` counter guarded by a
  thread lock (`api/budget.py`), explicitly relying on Cloud Run
  `max-instances=1` (ADR-0010) so there is a single writer.

That single-instance pin is the scaling ceiling. Run two instances and each
keeps its own limiter and its own daily counter, so a client's effective rate
limit and the global daily cap both multiply by the instance count - the
limits stop meaning what they say. Any real product traffic needs more than
one instance, so this state must move to a shared store before `max-instances`
can be lifted.

This must not regress the public demo or the local dev/test experience, which
today need no external services.

## Decision

1. **One `REDIS_URL` switch drives both backends.** A single `REDIS_URL`
   setting (default empty) is the scale-out toggle. Empty means in-memory
   everything (current behavior); set to `redis://<host>:<port>` it becomes the
   shared store for both the limiter and the daily cap. The limiter passes it as
   its slowapi/`limits` `storage_uri` (falling back to `memory://` when empty),
   so the `@limiter.limit` decorators and `key_func` (ADR-0023) are unchanged.
   One store, one config value, both concerns scale together.

2. **Introduce a `Budget` protocol for the daily cap** with two
   implementations selected by configuration:
   - `InMemoryDailyBudget` - the current `DailyBudget`, unchanged behavior,
     the default.
   - `RedisDailyBudget` - a shared counter using Redis `INCR` on a
     UTC-day-keyed field with a TTL set to expire at the next 00:00 UTC, so
     roll-over and cross-instance sharing both come from Redis semantics rather
     than in-process state.
   `DailyCapMiddleware` already depends only on `check_and_increment()`, so it
   is unaffected; the protocol just names the seam that already exists.

3. **Both default to in-memory; Redis is opt-in.** With no Redis URI
   configured, the service behaves exactly as today - zero external
   dependencies for the demo, local dev, and tests. Selecting Redis is a
   deployment choice, not a code change.

4. **The Redis client is an optional extra, not a core dependency.** Add a
   `redis` extra (`pip install -e '.[redis]'`), lazy-imported only when a Redis
   URI is configured - mirroring how `mcp`, `rerank`, and `observability` are
   handled (ADR-0021 etc.). The base install and the Cloud Run runtime image
   stay lightweight unless a deployment opts into scale-out. **This is the one
   dependency decision that needs explicit approval before implementation.**

5. **Lift `max-instances`** in the Cloud Run manifest only once the shared
   backends are in place and measured. The lift is the last step, gated on the
   measurement below, not part of the same change that adds the backends.

## Consequences

- The public demo can keep running single-instance with in-memory state and
  zero new infrastructure; nothing about ADR-0010's cost ceiling changes unless
  a deployment turns on Redis and raises `max-instances`.
- Testing the Redis path needs a fake or real Redis. Plan: add `fakeredis` as a
  dev dependency so the `RedisDailyBudget` and shared-limiter behavior are unit
  testable without a running server; optionally a Redis service container in CI
  for one integration test. (Second dependency decision, dev-only.)
- The measurement (governing rule): a test proving two `RedisDailyBudget`
  instances backed by the same store enforce one combined cap (not two), and a
  test proving two `Limiter`s sharing one storage backend enforce a single
  rate limit across both. Plus confirmation that with no Redis URI the behavior
  and test suite are byte-for-byte unchanged. Only after those pass does the
  `max-instances` lift land.
- `RedisDailyBudget` introduces a network dependency on the hot path for the
  cap check; a Redis outage must fail safe. Decision: on a Redis error the cap
  check fails open (allows the request) and logs, because a limiter/cap outage
  should degrade availability of governance, not of the service. This tradeoff
  is called out here so it is a conscious choice, revisitable if abuse risk
  outweighs availability.

## Alternatives considered

- **Express the daily cap as a `1/day` limit on a fixed global key via the same
  `limits` storage,** reusing the limiter backend and avoiding a second Redis
  code path. Attractive (one backend, one config) but rejected for now: the
  current cap has precise 00:00 UTC reset semantics and a `reset()` test seam,
  whereas `limits` fixed/rolling windows reset relative to first hit. Keeping an
  explicit `Budget` implementation preserves the exact semantics and keeps the
  cap legible. Worth revisiting if maintaining two Redis code paths proves
  costly.
- **Make Redis a hard core dependency.** Rejected: it would burden the demo and
  every local run with infrastructure they do not need, against the
  lightweight-base-install pattern the repo already follows.
- **Sticky sessions / instance affinity** to keep per-client state on one
  instance. Rejected: fragile, does not fix the global daily cap, and couples
  scaling to the load balancer.
