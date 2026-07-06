"""Integration tests for the global daily-cap middleware (ADR-0010)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_harness.api.server import app
from rag_harness.models import Chunk

client = TestClient(app)


def _mock_chunk() -> Chunk:
    return Chunk(
        id="docs/a.md::0",
        text="Some content.",
        source_file="docs/a.md",
        git_commit="abc123",
        doc_version="v1.29",
        chunk_index=0,
        heading_path=["A"],
    )


@pytest.fixture
def _tiny_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the daily cap to 2 for the test scope."""
    monkeypatch.setattr(app.state.daily_budget, "_cap", 2)
    app.state.daily_budget.reset()


def test_daily_cap_rejects_over_cap_with_429(_tiny_cap: None) -> None:
    """The (cap+1)th /query returns 429 with a demo_daily_limit_reached body."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_mock_chunk()])

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
    ):
        # First two hits succeed (cap=2)
        for i in range(2):
            r = client.post("/query", json={"question": "test?"})
            assert r.status_code == 200, f"hit {i + 1} unexpectedly rejected: {r.text}"

        # Third hit trips the daily cap
        r = client.post("/query", json={"question": "test?"})
        assert r.status_code == 429
        body = r.json()
        assert body["error_type"] == "demo_daily_limit_reached"
        assert "daily request cap" in body["message"].lower()
        assert body["detail"] is None


def test_daily_cap_does_not_count_health_or_ready(_tiny_cap: None) -> None:
    """/health and /ready must not consume budget slots — they are operator probes."""
    # Consume nothing on /health and /ready
    for _ in range(5):
        client.get("/health")
        client.get("/ready")
    # Full cap should still be available afterwards
    assert app.state.daily_budget.remaining() == 2


def test_daily_cap_does_not_count_metrics(_tiny_cap: None) -> None:
    """/metrics is a scrape endpoint — it must never consume budget slots."""
    for _ in range(5):
        client.get("/metrics")
    assert app.state.daily_budget.remaining() == 2
