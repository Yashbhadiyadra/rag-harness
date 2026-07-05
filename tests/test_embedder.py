"""Unit tests for the ingest embedder — usage recording, batching semantics."""

from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.models import Chunk
from rag_harness.observability.usage import collect_usage


def _chunk(cid: str, text: str = "some text") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        source_file=f"docs/{cid}.md",
        git_commit="abc123",
        doc_version="v1.32",
        chunk_index=0,
        heading_path=[],
    )


def _mock_response(embeddings: list[list[float]], prompt_tokens: int) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=vec) for vec in embeddings]
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=None)
    return resp


def test_embed_chunks_records_usage_once_per_batch() -> None:
    chunks = [_chunk(str(i), text=f"text {i}") for i in range(3)]

    with patch("rag_harness.ingest.embedder._client") as mock_client:
        mock_client.embeddings.create = AsyncMock(
            return_value=_mock_response([[0.1] * 4, [0.2] * 4, [0.3] * 4], prompt_tokens=30)
        )
        with collect_usage() as usage_list:
            from rag_harness.ingest.embedder import embed_chunks

            result = embed_chunks(chunks)

    assert len(result) == 3
    # Three chunks fit in one batch → one API call → one usage record
    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 30
    assert usage_list[0].output_tokens == 0


def test_embed_chunks_skips_api_and_records_no_usage_on_cache_hit() -> None:
    chunk = _chunk("a", text="cached text")

    cache = MagicMock()
    cache.get.return_value = [0.5, 0.5, 0.5, 0.5]  # cache hit

    with patch("rag_harness.ingest.embedder._client") as mock_client:
        mock_client.embeddings.create = AsyncMock()
        with collect_usage() as usage_list:
            from rag_harness.ingest.embedder import embed_chunks

            result = embed_chunks([chunk], cache=cache)

    assert len(result) == 1
    # Cache hit means no API call → no usage recorded
    assert len(usage_list) == 0
    mock_client.embeddings.create.assert_not_called()
