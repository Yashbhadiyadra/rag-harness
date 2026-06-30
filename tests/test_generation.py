from unittest.mock import MagicMock, patch

from rag_harness.generation.generator import generate
from rag_harness.models import Chunk


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


def test_generate_returns_answer() -> None:
    chunks = [_make_chunk("RoleBinding grants permissions to a user.")]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Use RoleBinding to grant permissions."

    with patch("rag_harness.generation.generator.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        answer = generate("How do I grant permissions in Kubernetes?", chunks)

    assert answer == "Use RoleBinding to grant permissions."


def test_generate_empty_chunks_returns_fallback() -> None:
    answer = generate("What is RBAC?", chunks=[])
    assert "not have enough information" in answer


def test_generate_uses_temperature_zero() -> None:
    chunks = [_make_chunk("Some content.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer."

    with patch("rag_harness.generation.generator.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        generate("A question?", chunks)
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0


def test_generate_includes_context_in_prompt() -> None:
    chunks = [_make_chunk("ClusterRole applies cluster-wide.")]
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Answer."

    with patch("rag_harness.generation.generator.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        generate("What is a ClusterRole?", chunks)
        call_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
        user_message = call_kwargs["messages"][1]["content"]
        assert "ClusterRole applies cluster-wide." in user_message
