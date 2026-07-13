"""Integration tests for the security-headers middleware."""

from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from rag_harness.api.server import app

client = TestClient(app)

_EXPECTED = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "no-referrer",
}


def _assert_security_headers(headers: httpx.Headers) -> None:
    # httpx Headers are case-insensitive - no dict() conversion (it lowercases keys)
    for name, value in _EXPECTED.items():
        assert headers.get(name) == value, f"missing or wrong {name}"
    csp = headers.get("Content-Security-Policy", "")
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_headers_on_health() -> None:
    """Plain JSON endpoints carry the full header set."""
    response = client.get("/health")
    assert response.status_code == 200
    _assert_security_headers(response.headers)


def test_headers_on_demo_ui() -> None:
    """The browser-facing demo UI page carries the full header set."""
    response = client.get("/")
    assert response.status_code == 200
    _assert_security_headers(response.headers)


def test_headers_on_static_assets() -> None:
    """Static assets are served with the header set too (nosniff matters here)."""
    response = client.get("/static/app.js")
    assert response.status_code == 200
    _assert_security_headers(response.headers)


def test_headers_on_validation_error() -> None:
    """Error responses carry the headers - the middleware wraps every path."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    _assert_security_headers(response.headers)


def test_headers_on_kill_switch_rejection() -> None:
    """Headers must be stamped even when an inner middleware short-circuits."""
    with patch("rag_harness.api.middleware.kill_switch.settings.demo_enabled", False):
        response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 503
    _assert_security_headers(response.headers)


def test_headers_on_404() -> None:
    """Unknown routes still get the header set."""
    response = client.get("/no-such-route")
    assert response.status_code == 404
    _assert_security_headers(response.headers)
