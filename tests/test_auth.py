"""Tests for API-key authentication (ADR-0023).

The measurement the governing rule requires: valid key -> 200, missing/bad key
-> 401, per-key rate-limit isolation, probes stay open, and enabling auth with
no keys fails at startup.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from rag_harness.api.auth import (
    _extract_bearer,
    hash_key,
    rate_limit_identity,
    verify_key,
)
from rag_harness.api.server import app
from rag_harness.config import Settings, settings
from rag_harness.models import Chunk

client = TestClient(app)

RAW_KEY = "test-secret-key-abc123"
KEY_HASH = hashlib.sha256(RAW_KEY.encode()).hexdigest()


def _make_chunk() -> Chunk:
    return Chunk(
        id="doc::0",
        text="RoleBinding grants permissions.",
        source_file="content/en/docs/security/rbac.md",
        git_commit="abc123",
        doc_version="v1.29",
        chunk_index=0,
        heading_path=["Security", "RBAC"],
    )


def _http_request(headers: dict[str, str]) -> Request:
    """Build a minimal ASGI Request with the given headers and a fixed client IP."""
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("203.0.113.7", 12345),
    }
    return Request(scope)


# --- pure helpers -----------------------------------------------------------


def test_hash_key_matches_sha256() -> None:
    assert hash_key(RAW_KEY) == KEY_HASH


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),  # scheme is case-insensitive
        ("Bearer   abc  ", "abc"),  # surrounding whitespace stripped
        ("Basic abc", None),  # wrong scheme
        ("abc", None),  # no scheme
        ("Bearer ", None),  # empty token
        ("", None),
        (None, None),
    ],
)
def test_extract_bearer(header: str | None, expected: str | None) -> None:
    assert _extract_bearer(header) == expected


def test_verify_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", KEY_HASH)
    assert verify_key(RAW_KEY) is True
    assert verify_key("wrong-key") is False


# --- dependency behaviour through /query ------------------------------------


def test_query_open_when_auth_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default posture: the public demo needs no key."""
    monkeypatch.setattr(settings, "api_auth_enabled", False)
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk()])
    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="An answer.",
        ),
    ):
        response = client.post("/query", json={"question": "How do I configure RBAC?"})
    assert response.status_code == 200


def test_query_401_when_auth_enabled_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", KEY_HASH)
    response = client.post("/query", json={"question": "q?"})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["error_type"] == "authentication_error"


def test_query_401_with_unknown_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", KEY_HASH)
    response = client.post(
        "/query",
        json={"question": "q?"},
        headers={"Authorization": "Bearer not-a-real-key"},
    )
    assert response.status_code == 401


def test_query_200_with_valid_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", KEY_HASH)
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk()])
    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="Use RoleBinding.",
        ),
    ):
        response = client.post(
            "/query",
            json={"question": "How do I configure RBAC?"},
            headers={"Authorization": f"Bearer {RAW_KEY}"},
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "Use RoleBinding."


def test_probes_stay_open_when_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Liveness/readiness/scrape endpoints never require a key."""
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", KEY_HASH)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code != 401  # 200 or 503 by env, never 401
    assert client.get("/metrics").status_code == 200


# --- rate-limit identity ----------------------------------------------------


def test_rate_limit_identity_is_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct valid keys get distinct, key-scoped buckets."""
    second_raw = "second-client-key"
    monkeypatch.setattr(settings, "api_keys", f"{KEY_HASH},{hash_key(second_raw)}")
    id1 = rate_limit_identity(_http_request({"Authorization": f"Bearer {RAW_KEY}"}))
    id2 = rate_limit_identity(_http_request({"Authorization": f"Bearer {second_raw}"}))
    assert id1.startswith("key:") and id2.startswith("key:")
    assert id1 != id2


def test_rate_limit_identity_falls_back_to_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anonymous or invalid-key traffic is bucketed by client IP."""
    monkeypatch.setattr(settings, "api_keys", KEY_HASH)
    assert rate_limit_identity(_http_request({})) == "203.0.113.7"
    bad = rate_limit_identity(_http_request({"Authorization": "Bearer nope"}))
    assert bad == "203.0.113.7"


def test_per_key_quota_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """One key exhausting its burst quota (429) does not affect a different key.

    End-to-end proof of the per-key rate limit: the 3/minute burst trips a 429
    for key A's fourth request, while key B - a distinct bucket - still gets 200.
    """
    second_raw = "second-client-key"
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    monkeypatch.setattr(settings, "api_keys", f"{KEY_HASH},{hash_key(second_raw)}")

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk()])
    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="An answer.",
        ),
    ):
        key_a = {"Authorization": f"Bearer {RAW_KEY}"}
        # Burst limit is 3/minute: three succeed, the fourth is throttled.
        for _ in range(3):
            assert client.post("/query", json={"question": "q?"}, headers=key_a).status_code == 200
        assert client.post("/query", json={"question": "q?"}, headers=key_a).status_code == 429

        # A different key has its own bucket and is unaffected by key A's throttle.
        key_b = {"Authorization": f"Bearer {second_raw}"}
        assert client.post("/query", json={"question": "q?"}, headers=key_b).status_code == 200


# --- fail-closed configuration ----------------------------------------------


def test_auth_enabled_without_keys_fails_at_startup() -> None:
    """Enabling auth with an empty allowlist is a startup error, not a silent open."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="x", api_auth_enabled=True, api_keys="")
