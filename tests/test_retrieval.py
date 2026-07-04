from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.observability.usage import collect_usage
from rag_harness.retrieval.dense import DenseRetriever


def _make_retriever() -> DenseRetriever:
    """Construct a DenseRetriever with all external clients mocked out."""
    with (
        patch("rag_harness.retrieval.dense.AsyncOpenAI"),
        patch("rag_harness.retrieval.dense.chromadb.PersistentClient"),
    ):
        return DenseRetriever()


def test_retrieve_returns_chunks() -> None:
    retriever = _make_retriever()

    embed_resp = MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
    retriever._openai.embeddings.create = AsyncMock(return_value=embed_resp)  # type: ignore[attr-defined]
    retriever._collection.query.return_value = {  # type: ignore[attr-defined]
        "ids": [["content/en/docs/security/rbac.md::0"]],
        "documents": [["RoleBinding grants permissions to users."]],
        "metadatas": [
            [
                {
                    "source_file": "content/en/docs/security/rbac.md",
                    "git_commit": "abc123",
                    "doc_version": "v1.29",
                    "chunk_index": "0",
                    "heading_path": '["Security", "RBAC"]',
                }
            ]
        ],
    }

    chunks = retriever.retrieve("How do I configure RBAC?", top_k=1)

    assert len(chunks) == 1
    assert chunks[0].source_file == "content/en/docs/security/rbac.md"
    assert chunks[0].git_commit == "abc123"
    assert chunks[0].heading_path == ["Security", "RBAC"]
    assert "RoleBinding" in chunks[0].text


def test_retrieve_uses_configured_top_k() -> None:
    retriever = _make_retriever()
    embed_resp = MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])])
    retriever._openai.embeddings.create = AsyncMock(return_value=embed_resp)  # type: ignore[attr-defined]
    retriever._collection.query.return_value = {  # type: ignore[attr-defined]
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }

    retriever.retrieve("some query", top_k=7)

    call_kwargs = retriever._collection.query.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["n_results"] == 7


def test_dense_retrieve_records_usage_inside_collect_block() -> None:
    retriever = _make_retriever()
    resp = MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
    resp.usage = MagicMock(prompt_tokens=8, completion_tokens=0)
    retriever._openai.embeddings.create = AsyncMock(return_value=resp)  # type: ignore[attr-defined]
    retriever._collection.query.return_value = {  # type: ignore[attr-defined]
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }

    with collect_usage() as usage_list:
        retriever.retrieve("q", top_k=3)

    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 8
    assert usage_list[0].output_tokens == 0
