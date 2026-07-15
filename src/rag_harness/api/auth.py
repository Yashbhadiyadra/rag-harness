"""API-key authentication: a hashed allowlist with no plaintext at rest (ADR-0023).

Authentication is opt-in via ``API_AUTH_ENABLED``; the public demo runs with it
off. When enabled, ``/query`` requires a valid key presented as
``Authorization: Bearer <key>``. Keys are compared as SHA-256 digests against
the ``API_KEYS`` allowlist, so no plaintext key is ever stored or logged.
"""

import hashlib
import hmac

from fastapi import Request
from slowapi.util import get_remote_address

from rag_harness.api.errors import AuthenticationError
from rag_harness.config import settings


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
    """Constant-time check that ``raw_key``'s hash is in the configured allowlist.

    Uses :func:`hmac.compare_digest` per candidate so comparison time does not
    leak how many leading characters of a digest matched.
    """
    candidate = hash_key(raw_key)
    return any(hmac.compare_digest(candidate, known) for known in settings.api_key_hashes)


async def require_api_key(request: Request) -> None:
    """FastAPI dependency enforcing a valid API key when auth is enabled.

    A no-op when ``API_AUTH_ENABLED`` is false (the public demo). Otherwise a
    missing or unknown key raises :class:`AuthenticationError`, which the
    server's error handler renders as ``401`` with a ``WWW-Authenticate:
    Bearer`` header.
    """
    if not settings.api_auth_enabled:
        return
    token = _extract_bearer(request.headers.get("Authorization"))
    if token is None or not verify_key(token):
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
