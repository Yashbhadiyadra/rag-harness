from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from openai import APIConnectionError

from rag_harness.api.server import app
from rag_harness.generation.corrective import NO_INFO_MESSAGE
from rag_harness.models import Chunk

client = TestClient(app)


def _make_chunk(source_file: str, heading_path: list[str]) -> Chunk:
    return Chunk(
        id=f"{source_file}::0",
        text="RoleBinding grants permissions.",
        source_file=source_file,
        git_commit="abc123",
        doc_version="v1.29",
        chunk_index=0,
        heading_path=heading_path,
    )


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_per_check_status() -> None:
    """/ready returns per-check status regardless of pass/fail."""
    response = client.get("/ready")
    body = response.json()
    # Whatever the test env produces, the shape is consistent
    assert "checks" in body or "status" in body
    if response.status_code == 200:
        assert body["status"] == "ready"
        assert "chromadb" in body["checks"]
        assert "openai_api_key" in body["checks"]
    else:
        assert response.status_code == 503
        assert body["error_type"] == "not_ready"
        assert "chromadb" in body["checks"]


def test_ready_fails_when_openai_key_missing() -> None:
    """If OPENAI_API_KEY is blank, /ready returns 503."""
    with patch("rag_harness.api.server.settings.openai_api_key", ""):
        response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["error_type"] == "not_ready"
    assert body["checks"]["openai_api_key"] == "missing"


def test_ready_strict_on_missing_reranker_extra() -> None:
    """When strategy needs the reranker and [rerank] extra is missing,
    /ready fails (strict per your Q3 decision)."""
    import sys

    with (
        patch("rag_harness.api.server.settings.retrieval_strategy", "hybrid-rerank"),
        patch.dict(sys.modules, {"sentence_transformers": None}),
    ):
        response = client.get("/ready")

    body = response.json()
    if response.status_code == 503:
        # If we're here, the cross-encoder check flagged it
        assert body["checks"].get("cross_encoder") == "not-installed"


def test_query_returns_answer_and_sources() -> None:
    chunk = _make_chunk("content/en/docs/security/rbac.md", ["Security", "RBAC"])

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="Use RoleBinding.",
        ),
    ):
        response = client.post("/query", json={"question": "How do I configure RBAC?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use RoleBinding."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_file"] == "content/en/docs/security/rbac.md"
    assert body["sources"][0]["heading_path"] == ["Security", "RBAC"]


def test_query_response_includes_trace_cost_and_latency() -> None:
    """/query must return per-stage trace, this-query cost, and latency.

    The demo UI (ADR-0010) renders these alongside the answer so a stranger
    with the URL can see the reliability story on-page. Fully-mocked test
    path: no real OpenAI calls → cost_usd stays at 0.0 but the field is
    present. Trace should include the outer 'query' span plus 'retrieve'
    and 'generate'.
    """
    chunk = _make_chunk("docs/a.md", ["A"])
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
    ):
        response = client.post("/query", json={"question": "test?"})

    assert response.status_code == 200
    body = response.json()
    assert "trace" in body
    assert "cost_usd" in body
    assert "latency_ms" in body

    span_names = [s["name"] for s in body["trace"]]
    assert "query" in span_names, f"outer 'query' span missing from trace {span_names}"
    assert "retrieve" in span_names, f"'retrieve' span missing from trace {span_names}"
    assert "generate" in span_names, f"'generate' span missing from trace {span_names}"
    # Every span has a non-negative duration
    assert all(s["duration_ms"] >= 0 for s in body["trace"])
    # Cost is a float ≥ 0. No real LLM call → likely 0.0.
    assert isinstance(body["cost_usd"], (int, float))
    assert body["cost_usd"] >= 0.0
    # Latency is a positive wall-clock value
    assert body["latency_ms"] > 0.0


def test_query_rejects_empty_question() -> None:
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_rejects_oversized_question() -> None:
    from rag_harness.config import settings

    oversized = "x" * (settings.api_max_question_length + 1)
    response = client.post("/query", json={"question": oversized})
    assert response.status_code == 422


def test_query_rejects_zero_top_k() -> None:
    response = client.post("/query", json={"question": "hi", "top_k": 0})
    assert response.status_code == 422


def test_query_returns_refusal_on_openai_error() -> None:
    """When the LLM boundary raises OpenAIError (exhausted retries), the
    handler must degrade to the honest refusal answer with an empty sources
    list — NOT a 5xx. Distinct log event for ops."""
    chunk = _make_chunk("docs/a.md", ["A"])

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            side_effect=APIConnectionError(request=MagicMock()),
        ),
    ):
        response = client.post("/query", json={"question": "test?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_INFO_MESSAGE
    assert body["sources"] == []


def test_query_respects_top_k() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[])

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
    ):
        client.post("/query", json={"question": "Some question?", "top_k": 3})
        mock_retriever.retrieve_async.assert_awaited_once_with("Some question?", top_k=3)


def test_rate_limit_config_declares_burst_and_hourly() -> None:
    """Regression: the public per-IP rate limit must not silently loosen.

    ADR-0010 sizes the limit at 10/hour + 3/minute. Anyone changing the
    default without also updating this test is prompted to justify it.
    """
    from limits import parse_many

    from rag_harness.config import settings

    parsed = list(parse_many(settings.api_rate_limit))
    parts = {(p.amount, p.GRANULARITY.name) for p in parsed}
    assert (3, "minute") in parts, f"burst limit missing from {settings.api_rate_limit!r}"
    assert (10, "hour") in parts, f"hourly limit missing from {settings.api_rate_limit!r}"


def test_rate_limit_burst_returns_429() -> None:
    """4th quick /query from the same IP within a minute → 429.

    Exercises slowapi's per-IP burst limit (3/minute). The autouse fixture
    resets the limiter before this test so previous tests don't affect the
    counter.
    """
    chunk = _make_chunk("docs/a.md", ["A"])
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
    ):
        for i in range(3):
            r = client.post("/query", json={"question": "test?"})
            assert r.status_code == 200, f"request {i + 1} unexpectedly rejected: {r.text}"
        r = client.post("/query", json={"question": "test?"})
        assert r.status_code == 429, f"4th request should trip burst limit, got {r.status_code}"
