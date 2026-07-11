"""Integration tests for the demo kill-switch middleware (ADR-0010)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_harness.api.server import app

client = TestClient(app)


def test_kill_switch_off_query_returns_503() -> None:
    """When DEMO_ENABLED=false, /query returns 503 with a demo_disabled body."""
    with patch("rag_harness.api.middleware.kill_switch.settings.demo_enabled", False):
        response = client.post("/query", json={"question": "anything"})

    assert response.status_code == 503
    body = response.json()
    assert body["error_type"] == "demo_disabled"
    assert "disabled" in body["message"].lower()
    assert body["detail"] is None


def test_kill_switch_off_health_still_ok() -> None:
    """Kill switch must not silence /health - operators need liveness."""
    with patch("rag_harness.api.middleware.kill_switch.settings.demo_enabled", False):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_kill_switch_off_ready_still_reachable() -> None:
    """Kill switch must not silence /ready - operators need dependency status."""
    with patch("rag_harness.api.middleware.kill_switch.settings.demo_enabled", False):
        response = client.get("/ready")
    # /ready may be 200 or 503 depending on the test env; either is fine -
    # the point is the kill switch didn't hijack the response.
    assert response.status_code in (200, 503)
    body = response.json()
    if response.status_code == 200:
        assert body["status"] == "ready"
    else:
        assert body["error_type"] == "not_ready"


def test_kill_switch_off_metrics_still_reachable() -> None:
    """Kill switch must not silence /metrics - Prometheus needs to scrape."""
    with patch("rag_harness.api.middleware.kill_switch.settings.demo_enabled", False):
        response = client.get("/metrics")
    assert response.status_code == 200
    assert b"rag_query_total" in response.content


def test_kill_switch_on_query_reaches_handler() -> None:
    """Sanity check: with the kill switch OFF (default), /query is not blocked
    by the switch. Body validation may still 422 an empty request, but the
    response must not be the 503 demo_disabled body."""
    response = client.post("/query", json={"question": "   "})
    if response.status_code == 503:
        # If we ever get 503, it must be for a real reason, not the kill switch
        body = response.json()
        assert body["error_type"] != "demo_disabled"
