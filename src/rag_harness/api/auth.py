"""API-key authentication: a hashed allowlist with no plaintext at rest (ADR-0023).

Authentication is opt-in via ``API_AUTH_ENABLED``; the public demo runs with it
off. When enabled, ``/query`` requires a valid key presented as
``Authorization: Bearer <key>``. Keys are compared as SHA-256 digests against
the ``API_KEYS`` allowlist, so no plaintext key is ever stored or logged.
"""

import hashlib
import hmac
from dataclasses import dataclass

from fastapi import Request
from slowapi.util import get_remote_address

from rag_harness.api.errors import AuthenticationError
from rag_harness.config import settings


@dataclass(frozen=True)
class TenantContext:
    """The resolved identity of a request: which tenant, and its corpus (ADR-0025)."""

    tenant_id: str
    collection: str


def default_tenant() -> TenantContext:
    """The single-tenant context: the default collection.

    Used when auth is off (public demo) and for flat-allowlist keys that are
    not assigned to a named tenant. Reads ``settings.chroma_collection`` at call
    time so test overrides are respected.
    """
    return TenantContext(tenant_id="default", collection=settings.chroma_collection)


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a raw API key.

    This is the only function that touches a plaintext key. The digest is what
    goes in the ``API_KEYS`` allowlist; the raw key is never persisted.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _extract_bearer(authorization: str | None) -> str | None:
    """Return the token from an ``Authorization: Bearer <token>`` header.

    Returns ``None`` for a missing header, a non-Bearer scheme, or an empty
    token so callers can treat "no usable credential" uniformly.
    """
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def verify_key(raw_key: str) -> bool:
    """Constant-time check that ``raw_key``'s hash is an accepted key.

    Accepts any valid key of any tenant plus the flat allowlist. Uses
    :func:`hmac.compare_digest` per candidate so comparison time does not leak
    how many leading characters of a digest matched.
    """
    candidate = hash_key(raw_key)
    return any(hmac.compare_digest(candidate, known) for known in settings.all_key_hashes)


def resolve_tenant(raw_key: str) -> TenantContext | None:
    """Map a raw API key to its tenant context, or None if the key is unknown.

    A key listed under a named tenant resolves to that tenant's collection; a
    key only in the flat allowlist resolves to the default tenant. Named
    tenants are checked first (ADR-0025).
    """
    candidate = hash_key(raw_key)
    for tenant_id, spec in settings.tenants.items():
        if any(hmac.compare_digest(candidate, known) for known in spec.key_hashes):
            return TenantContext(tenant_id=tenant_id, collection=spec.collection)
    for known in settings.api_key_hashes:
        if hmac.compare_digest(candidate, known):
            return default_tenant()
    return None


async def require_api_key(request: Request) -> TenantContext:
    """FastAPI dependency resolving the request's tenant, enforcing auth.

    Returns the default tenant when ``API_AUTH_ENABLED`` is false (the public
    demo). Otherwise a missing or unknown key raises :class:`AuthenticationError`
    (401 with ``WWW-Authenticate: Bearer``); a valid key returns its resolved
    :class:`TenantContext`, which ``/query`` uses to select the tenant's corpus.
    """
    if not settings.api_auth_enabled:
        return default_tenant()
    token = _extract_bearer(request.headers.get("Authorization"))
    if token is not None:
        tenant = resolve_tenant(token)
        if tenant is not None:
            return tenant
    raise AuthenticationError("valid API key required")


def rate_limit_identity(request: Request) -> str:
    """Rate-limit bucket key: per-API-key when authenticated, else per-IP.

    Parses and validates the header here rather than reading state an endpoint
    dependency sets, so the bucket is independent of dependency-resolution order
    relative to the slowapi limiter. Authenticated callers get their own fair
    quota (correct behind shared NAT/proxies); anonymous traffic stays bounded
    by client IP. The bucket key is the key's hash, never the raw key.
    """
    token = _extract_bearer(request.headers.get("Authorization"))
    if token is not None and verify_key(token):
        return f"key:{hash_key(token)}"
    return get_remote_address(request)
