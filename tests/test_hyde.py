"""Unit tests for the HyDE query transformation retriever."""

from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.models import Chunk
from rag_harness.observability.usage import collect_usage
from rag_harness.retrieval.hyde import HyDERetriever


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        text=f"text for {cid}",
        source_file=f"docs/{cid}.md",
        git_commit="abc123",
        doc_version="v1.32",
        chunk_index=0,
        heading_path=[],
    )


def _mock_openai_returning(
    hypothesis: str, prompt_tokens: int = 30, completion_tokens: int = 20
) -> MagicMock:
    """Return an AsyncOpenAI mock whose async .create returns *hypothesis*."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = hypothesis
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _mock_base_retriever(return_value: list[Chunk]) -> MagicMock:
    """Return a mock base retriever with retrieve_async as AsyncMock."""
    base = MagicMock()
    base.retrieve_async = AsyncMock(return_value=return_value)
    return base


def test_hyde_passes_hypothesis_to_base_retriever() -> None:
    base = _mock_base_retriever([_chunk("a")])

    with patch("rag_harness.retrieval.hyde.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _mock_openai_returning(
            "A PodDisruptionBudget limits how many pods can be down during voluntary disruptions."
        )
        hyde = HyDERetriever(base_retriever=base)
        hyde.retrieve("How do I keep pods available during upgrades?", top_k=5)

    # The base retriever should receive the HYPOTHESIS, not the raw query
    call_args = base.retrieve_async.await_args
    passed_query = call_args[0][0] if call_args[0] else call_args[1]["query"]
    assert "PodDisruptionBudget" in passed_query


def test_hyde_falls_back_to_raw_query_on_empty_hypothesis() -> None:
    base = _mock_base_retriever([_chunk("a")])

    with patch("rag_harness.retrieval.hyde.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _mock_openai_returning("")
        hyde = HyDERetriever(base_retriever=base)
        hyde.retrieve("what is a pod?", top_k=5)

    call_args = base.retrieve_async.await_args
    passed_query = call_args[0][0] if call_args[0] else call_args[1]["query"]
    assert passed_query == "what is a pod?"


def test_hyde_falls_back_to_raw_query_on_openai_error() -> None:
    base = _mock_base_retriever([_chunk("a")])

    with patch("rag_harness.retrieval.hyde.AsyncOpenAI") as mock_openai_cls:
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))
        mock_openai_cls.return_value = client
        hyde = HyDERetriever(base_retriever=base)
        hyde.retrieve("what is a pod?", top_k=5)

    call_args = base.retrieve_async.await_args
    passed_query = call_args[0][0] if call_args[0] else call_args[1]["query"]
    assert passed_query == "what is a pod?"


def test_hyde_forwards_top_k_to_base() -> None:
    base = _mock_base_retriever([])

    with patch("rag_harness.retrieval.hyde.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _mock_openai_returning("hypothesis text")
        hyde = HyDERetriever(base_retriever=base)
        hyde.retrieve("query", top_k=7)

    base.retrieve_async.assert_awaited_once_with("hypothesis text", top_k=7)


def test_hyde_calls_llm_with_configured_model() -> None:
    base = _mock_base_retriever([])

    with patch("rag_harness.retrieval.hyde.AsyncOpenAI") as mock_openai_cls:
        client = _mock_openai_returning("hypothesis")
        mock_openai_cls.return_value = client
        hyde = HyDERetriever(base_retriever=base, model="test-model-x")
        hyde.retrieve("query")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model-x"
    assert kwargs["temperature"] == 0.0


def test_hyde_records_usage_inside_collect_block() -> None:
    base = _mock_base_retriever([])

    with patch("rag_harness.retrieval.hyde.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = _mock_openai_returning(
            "hypothesis", prompt_tokens=25, completion_tokens=15
        )
        with collect_usage() as usage_list:
            hyde = HyDERetriever(base_retriever=base)
            hyde.retrieve("query")

    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 25
    assert usage_list[0].output_tokens == 15
