import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.generation.generator import generate, generate_stream
from rag_harness.models import Chunk
from rag_harness.observability.usage import collect_usage


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        id="content/en/docs/security/rbac.md::0",
        text=text,
        source_file="content/en/docs/security/rbac.md",
        git_commit="abc123",
        doc_version="v1.29",
        chunk_index=0,
        heading_path=["Security", "RBAC"],
    )


def _mock_client_returning(mock_response: MagicMock) -> MagicMock:
    """Return a mock module-level _client whose async .create returns *mock_response*."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=mock_response)
    return client


def _make_stream_chunk(content: str | None = None, usage: MagicMock | None = None) -> MagicMock:
    """A single streaming chunk. content=None -> usage-only final chunk (no choices)."""
    chunk = MagicMock()
    if content is None:
        chunk.choices = []
    else:
        choice = MagicMock()
        choice.delta.content = content
        chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _mock_streaming_client(deltas: list[str], usage: MagicMock | None = None) -> MagicMock:
    """A mock _client whose awaited .create yields *deltas* then a usage-only chunk."""

    async def _stream() -> AsyncIterator[MagicMock]:
        for d in deltas:
            yield _make_stream_chunk(content=d)
        yield _make_stream_chunk(content=None, usage=usage)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_stream())
    return client


def _collect_stream(query: str, chunks: list[Chunk], client: MagicMock) -> list[str]:
    async def _run() -> list[str]:
        out: list[str] = []
        with patch("rag_harness.generation.generator._client", client):
            async for delta in generate_stream(query, chunks):
                out.append(delta)
        return out

    return asyncio.run(_run())


def test_generate_stream_yields_deltas_in_order() -> None:
    chunks = [_make_chunk("RoleBinding grants permissions to a user.")]
    client = _mock_streaming_client(["Use ", "RoleBinding", " to grant [1]."])

    deltas = _collect_stream("How do I grant permissions?", chunks, client)

    assert deltas == ["Use ", "RoleBinding", " to grant [1]."]
    assert "".join(deltas) == "Use RoleBinding to grant [1]."


def test_generate_stream_empty_chunks_yields_fallback() -> None:
    deltas = _collect_stream("What is RBAC?", [], MagicMock())
    assert len(deltas) == 1
    assert "not have enough information" in deltas[0]


def test_generate_stream_requests_streaming_with_usage() -> None:
    chunks = [_make_chunk("Content.")]
    client = _mock_streaming_client(["Answer."])

    _collect_stream("Q?", chunks, client)

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True
    assert call_kwargs["stream_options"] == {"include_usage": True}
    assert call_kwargs["temperature"] == 0


def test_generate_stream_records_usage_from_final_chunk() -> None:
    chunks = [_make_chunk("Content.")]
    client = _mock_streaming_client(
        ["Answer."], usage=MagicMock(prompt_tokens=42, completion_tokens=7)
    )

    async def _run() -> None:
        with patch("rag_harness.generation.generator._client", client):
            async for _ in generate_stream("Q?", chunks):
                pass

    with collect_usage() as usage_list:
        asyncio.run(_run())

    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 42
    assert usage_list[0].output_tokens == 7


def test_generate_returns_answer() -> None:
    chunks = [_make_chunk("RoleBinding grants permissions to a user.")]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Use RoleBinding to grant permissions."

    with patch("rag_harness.generation.generator._client", _mock_client_returning(mock_response)):
        answer = generate("How do I grant permissions in Kubernetes?", chunks)

    assert answer == "Use RoleBinding to grant permissions."


def test_generate_empty_chunks_returns_fallback() -> None:
    answer = generate("What is RBAC?", chunks=[])
    assert "not have enough information" in answer


def test_generate_empty_content_returns_fallback() -> None:
    # A content-filtered completion returns None content; do not emit "".
    chunks = [_make_chunk("Content.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None

    with patch("rag_harness.generation.generator._client", _mock_client_returning(mock_response)):
        answer = generate("Q?", chunks)

    assert "not have enough information" in answer


def test_generate_no_choices_returns_fallback() -> None:
    # An empty choices list must not IndexError - fall back to the refusal.
    chunks = [_make_chunk("Content.")]
    mock_response = MagicMock()
    mock_response.choices = []

    with patch("rag_harness.generation.generator._client", _mock_client_returning(mock_response)):
        answer = generate("Q?", chunks)

    assert "not have enough information" in answer


def test_generate_uses_temperature_zero() -> None:
    chunks = [_make_chunk("Some content.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer."
    client = _mock_client_returning(mock_response)

    with patch("rag_harness.generation.generator._client", client):
        generate("A question?", chunks)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0


def test_generate_includes_context_in_prompt() -> None:
    chunks = [_make_chunk("ClusterRole applies cluster-wide.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer."
    client = _mock_client_returning(mock_response)

    with patch("rag_harness.generation.generator._client", client):
        generate("What is a ClusterRole?", chunks)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        user_message = call_kwargs["messages"][1]["content"]
        assert "ClusterRole applies cluster-wide." in user_message


def test_generate_records_usage_inside_collect_block() -> None:
    chunks = [_make_chunk("Content.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer."
    mock_response.usage = MagicMock(prompt_tokens=42, completion_tokens=7)

    with (
        patch("rag_harness.generation.generator._client", _mock_client_returning(mock_response)),
        collect_usage() as usage_list,
    ):
        generate("Q?", chunks)

    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 42
    assert usage_list[0].output_tokens == 7


def test_generate_sets_genai_span_attributes() -> None:
    chunks = [_make_chunk("Content.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer."
    mock_response.usage = MagicMock(prompt_tokens=42, completion_tokens=7)

    with (
        patch("rag_harness.generation.generator._client", _mock_client_returning(mock_response)),
        patch("rag_harness.generation.generator.set_current_genai_attributes") as mock_semconv,
    ):
        generate("Q?", chunks)

    # The LLM boundary annotates the active span with GenAI attributes,
    # including the real token counts from the response.
    mock_semconv.assert_called_once()
    args = mock_semconv.call_args.args
    assert args[0] == "chat"
    assert args[2] == 42 and args[3] == 7
