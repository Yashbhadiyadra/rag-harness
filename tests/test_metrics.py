"""Unit tests for the Prometheus /metrics endpoint and counters."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from rag_harness.api.metrics import (
    QUERY_COST_USD,
    QUERY_TOTAL,
    prometheus_response,
)
from rag_harness.api.server import app
from rag_harness.models import Chunk

client = TestClient(app)


def _make_chunk() -> Chunk:
    return Chunk(
        id="docs/a.md::0",
        text="text",
        source_file="docs/a.md",
        git_commit="abc",
        doc_version="v1.32",
        chunk_index=0,
        heading_path=[],
    )


def test_metrics_endpoint_returns_prometheus_content_type() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=" in response.headers["content-type"]


def test_metrics_body_is_parseable_prometheus_format() -> None:
    body, _ = prometheus_response()
    # If the format is malformed, text_string_to_metric_families raises
    families = list(text_string_to_metric_families(body.decode("utf-8")))
    assert len(families) > 0
    family_names = {f.name for f in families}
    # The RAG-specific families should be present
    assert "rag_query" in family_names
    assert "rag_query_latency_seconds" in family_names


def test_metrics_excludes_default_process_and_platform_families() -> None:
    body, _ = prometheus_response()
    text = body.decode("utf-8")
    # Default prometheus_client collectors would emit these - we disabled them
    assert "process_cpu_seconds_total" not in text
    assert "python_gc_" not in text
    assert "python_info" not in text


def test_query_counter_increments_on_successful_query() -> None:
    chunk = _make_chunk()
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    before = QUERY_TOTAL.labels(strategy="dense", corrective="false")._value.get()

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch(
            "rag_harness.api.server.generate_async",
            new_callable=AsyncMock,
            return_value="answer",
        ),
    ):
        response = client.post("/query", json={"question": "test?"})

    assert response.status_code == 200
    after = QUERY_TOTAL.labels(strategy="dense", corrective="false")._value.get()
    assert after == before + 1


def test_query_cost_counter_increments_from_recorded_usage() -> None:
    chunk = _make_chunk()
    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[chunk])

    before = QUERY_COST_USD._value.get()

    from rag_harness.observability.usage import TokenUsage, record_usage

    async def _fake_generate(question: str, chunks: list) -> str:
        record_usage(TokenUsage("gpt-4o-mini", 100, 50, 0.045))
        return "answer"

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch("rag_harness.api.server.generate_async", side_effect=_fake_generate),
    ):
        client.post("/query", json={"question": "test?"})

    after = QUERY_COST_USD._value.get()
    # One call with cost 0.045 → counter grew by that amount
    assert round(after - before, 6) == 0.045


def test_error_counter_increments_when_retriever_raises() -> None:
    from rag_harness.api.metrics import QUERY_ERRORS_TOTAL

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(side_effect=RuntimeError("boom"))

    before = QUERY_ERRORS_TOTAL.labels(strategy="dense", error_type="RuntimeError")._value.get()

    # FastAPI TestClient re-raises server exceptions by default; the error
    # counter increments in the handler's `except` block before the exception
    # propagates. We verify that side effect regardless of how the client
    # surfaces the error.
    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        pytest.raises(RuntimeError, match="boom"),
    ):
        client.post("/query", json={"question": "test?"})

    after = QUERY_ERRORS_TOTAL.labels(strategy="dense", error_type="RuntimeError")._value.get()
    assert after == before + 1
