from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_harness.api.server import app
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


def test_query_returns_answer_and_sources() -> None:
    chunk = _make_chunk("content/en/docs/security/rbac.md", ["Security", "RBAC"])

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [chunk]

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch("rag_harness.api.server.generate", return_value="Use RoleBinding."),
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


def test_query_respects_top_k() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with (
        patch("rag_harness.api.server._get_retriever", return_value=mock_retriever),
        patch("rag_harness.api.server.generate", return_value="Answer."),
    ):
        client.post("/query", json={"question": "Some question?", "top_k": 3})
        mock_retriever.retrieve.assert_called_once_with("Some question?", top_k=3)
