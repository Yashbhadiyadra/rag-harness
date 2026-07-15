# ADR-0025: Multi-tenant corpus isolation

Date: 2026-07-15
Status: Accepted

## Context

Authentication (ADR-0023) established a caller identity - an API key - but
every valid key still queries the same single corpus in one Chroma collection
(`settings.chroma_collection`). To be a product, different tenants must see only
their own documents: a query authenticated as tenant A must never retrieve
tenant B's chunks.

Two facts shape the design:

- Every retrieval strategy composes down to `DenseRetriever`, which reads
  `settings.chroma_collection` directly. Isolation therefore means threading a
  per-tenant collection name through the retriever build path, which today is a
  process-global singleton keyed only on strategy.
- Bring-your-own-corpus (ADR-0019) already parameterizes ingest by corpus and
  collection, so per-tenant collections can be created with the existing
  machinery - no new ingest surface is needed to provision a tenant.

This phase delivers the isolation *guarantee*, not self-service onboarding.

## Decision

1. **Tenant identity comes from the API key via a config-driven registry.** A
   new `TENANTS` setting (JSON) maps each tenant id to its API-key hashes and
   its Chroma collection:
   `{"acme": {"key_hashes": ["<sha256>"], "collection": "tenant_acme"}}`.
   This extends the Phase 1 hashed allowlist (ADR-0023) without a database,
   and stays swappable for a DB-backed store later. Raw keys are never stored;
   only hashes, exactly as in ADR-0023.

2. **A per-tenant Chroma collection is the isolation boundary.** Isolation is
   by construction - separate collections, not a shared collection filtered by
   a tenant field. A separate collection cannot leak another tenant's data
   through a forgotten filter; it is defense in depth and matches Chroma's
   model. Retrieval threads a collection name through `build_retriever`;
   `DenseRetriever` gains a `collection_name` argument defaulting to
   `settings.chroma_collection` so existing single-tenant behavior is unchanged.

3. **Corpora are operator-provisioned.** An operator ingests each tenant's docs
   into that tenant's collection using the existing BYO-corpus ingest
   (ADR-0019). There is no upload endpoint in this phase - self-service
   provisioning is a separate product surface (file upload, async jobs,
   per-tenant storage and quotas, a large abuse surface) deferred until
   isolation exists and is proven.

4. **No silent fallback for named tenants.** A named tenant must have an
   explicit, existing collection. If its collection is missing or
   misconfigured, `/query` errors rather than serving the default corpus or
   another tenant's - a fallback would be a data-leak. Only the *default
   tenant* uses `settings.chroma_collection`.

5. **Back-compatible by default.** With auth off (the public demo) there is a
   single default tenant on the default collection - today's behavior,
   unchanged. A Phase 1 flat-allowlist key (`API_KEYS`, no tenant entry) also
   resolves to the default tenant. Multi-tenancy only engages when `TENANTS`
   is configured and auth is on.

6. **A key maps to exactly one identity.** Key hashes must be disjoint across
   tenants and the flat allowlist; overlap is a configuration error caught at
   startup, so a key can never resolve ambiguously to two tenants.

## Consequences

- `require_api_key` (ADR-0023) evolves from returning nothing to resolving a
  tenant context (id + collection) from the key and attaching it to the
  request; `/query` selects the tenant's collection. `verify_key` and the
  per-key rate-limit identity (ADR-0023) are unchanged - any valid key of any
  tenant still authenticates and is metered per key.
- The retriever singleton becomes a small cache keyed by
  `(strategy, collection)`, so each tenant gets its own retriever built lazily
  and reused. `DenseRetriever` and the strategies that compose it accept the
  collection name.
- The measurement (governing rule): an isolation test proving a query
  authenticated as tenant A returns only A's sources and never B's, across
  two real collections with disjoint documents; a test that the default tenant
  (auth off) behaves byte-for-byte as today; a test that a named tenant with a
  missing collection errors instead of leaking the default corpus; and a
  startup-validation test rejecting overlapping key hashes.
- Operators get a tenant-onboarding runbook in `deploy/`: hash the tenant's
  keys, ingest their corpus into a named collection, add the `TENANTS` entry.
- Not covered, deferred by design: DB-backed tenant store, self-service
  upload, per-tenant rate-limit tiers (Phase 2's per-key limiter already meters
  per key; tenant-level quotas can build on it), and per-tenant eval/golden
  sets (judge reliability is not assumed to transfer across corpora, per
  ADR-0022).

## Alternatives considered

- **One shared collection filtered by a `tenant` metadata field.** Rejected:
  a single missing or wrong filter on any query path leaks every tenant's data.
  Separate collections are isolation by construction; the cost (more
  collections) is negligible in Chroma.
- **DB-backed tenant registry now.** Rejected for this phase: there is no
  persistence layer for it, and a config registry is consistent with how
  ADR-0023 handles keys. Revisit with self-service onboarding.
- **Self-service corpus upload now.** Rejected: a large new product and
  security surface. Isolation is the prerequisite and must land and be proven
  first.
