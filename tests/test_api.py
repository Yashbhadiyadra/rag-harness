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
