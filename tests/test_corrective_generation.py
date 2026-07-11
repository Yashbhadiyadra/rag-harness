"""Unit tests for the corrective retrieve → critique → route flow."""

from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.generation.corrective import (
    NO_INFO_MESSAGE,
    CorrectiveResult,
    corrective_generate,
)
from rag_harness.generation.critic import Category
from rag_harness.models import Chunk
from rag_harness.observability.usage import collect_usage


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


def _mock_critic(scores: list[float], category: Category) -> MagicMock:
    """Mock critic with an async score_batch_async and sync categorise."""
    m = MagicMock()
    m.score_batch_async = AsyncMock(return_value=scores)
    m.categorise.return_value = category
    m._incorrect_threshold = 0.3
    return m


def _mock_openai_returning(
    text: str, prompt_tokens: int = 40, completion_tokens: int = 10
) -> MagicMock:
    """Return a mock AsyncOpenAI client whose async .create returns *text*."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# --- Correct path ---


def test_correct_category_filters_and_generates() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("a"), _chunk("b"), _chunk("c")])
    critic = _mock_critic([0.9, 0.5, 0.1], Category.CORRECT)  # c is below 0.3 threshold

    with (
        patch(
            "rag_harness.generation.corrective.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
        patch("rag_harness.generation.corrective.build_async_client"),
    ):
        result = corrective_generate("query", retriever, critic=critic, max_retries=0)

    assert isinstance(result, CorrectiveResult)
    assert result.answer == "Answer."
    assert result.category is Category.CORRECT
    assert result.attempts == 1
    # c dropped - it was below the incorrect threshold
    assert {c.id for c in result.chunks_used} == {"a", "b"}


# --- Ambiguous path ---


def test_ambiguous_category_generates_with_surviving_chunks() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("a"), _chunk("b")])
    critic = _mock_critic([0.5, 0.2], Category.AMBIGUOUS)

    with (
        patch(
            "rag_harness.generation.corrective.generate_async",
            new_callable=AsyncMock,
            return_value="Maybe answer.",
        ) as gen_mock,
        patch("rag_harness.generation.corrective.build_async_client"),
    ):
        result = corrective_generate("query", retriever, critic=critic, max_retries=0)

    assert result.category is Category.AMBIGUOUS
    # b dropped - 0.2 is below 0.3 incorrect threshold
    assert [c.id for c in result.chunks_used] == ["a"]
    gen_mock.assert_awaited_once_with("query", [_chunk("a")])


# --- Incorrect path with retry ---


def test_incorrect_triggers_reformulation_and_retry() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(
        side_effect=[
            [_chunk("bad1"), _chunk("bad2")],  # first attempt
            [_chunk("good")],  # after reformulation
        ]
    )

    call_count = {"n": 0}

    def _categorise(scores: list[float]) -> Category:
        call_count["n"] += 1
        # First attempt Incorrect, second attempt Correct
        return Category.INCORRECT if call_count["n"] == 1 else Category.CORRECT

    critic = MagicMock()
    critic.score_batch_async = AsyncMock(side_effect=[[0.1, 0.1], [0.9]])
    critic.categorise.side_effect = _categorise
    critic._incorrect_threshold = 0.3

    with (
        patch(
            "rag_harness.generation.corrective.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
        patch("rag_harness.generation.corrective.build_async_client") as mock_openai_cls,
    ):
        mock_openai_cls.return_value = _mock_openai_returning("What is a Pod controller?")
        result = corrective_generate("query", retriever, critic=critic, max_retries=1)

    assert result.answer == "Answer."
    assert result.category is Category.CORRECT
    assert result.attempts == 2
    assert result.reformulated_query == "What is a Pod controller?"
    assert [c.id for c in result.chunks_used] == ["good"]


# --- Incorrect path with no successful retry ---


def test_all_incorrect_returns_refusal_message() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("bad")])
    critic = _mock_critic([0.1], Category.INCORRECT)

    with (
        patch("rag_harness.generation.corrective.generate_async", new_callable=AsyncMock),
        patch("rag_harness.generation.corrective.build_async_client") as mock_openai_cls,
    ):
        mock_openai_cls.return_value = _mock_openai_returning("rewritten query")
        result = corrective_generate("query", retriever, critic=critic, max_retries=1)

    assert result.answer == NO_INFO_MESSAGE
    assert result.chunks_used == []
    assert result.category is Category.INCORRECT
    assert result.attempts == 2  # original + 1 retry


def test_no_retries_returns_refusal_immediately_on_incorrect() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("bad")])
    critic = _mock_critic([0.1], Category.INCORRECT)

    with (
        patch("rag_harness.generation.corrective.generate_async", new_callable=AsyncMock),
        patch("rag_harness.generation.corrective.build_async_client"),
    ):
        result = corrective_generate("query", retriever, critic=critic, max_retries=0)

    assert result.answer == NO_INFO_MESSAGE
    assert result.attempts == 1


# --- Robustness ---


def test_critic_scores_against_original_query_after_reformulation() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(
        side_effect=[
            [_chunk("bad")],
            [_chunk("good")],
        ]
    )
    critic = MagicMock()
    critic.score_batch_async = AsyncMock(side_effect=[[0.1], [0.9]])
    critic.categorise.side_effect = [Category.INCORRECT, Category.CORRECT]
    critic._incorrect_threshold = 0.3

    with (
        patch(
            "rag_harness.generation.corrective.generate_async",
            new_callable=AsyncMock,
            return_value="Answer.",
        ),
        patch("rag_harness.generation.corrective.build_async_client") as mock_openai_cls,
    ):
        mock_openai_cls.return_value = _mock_openai_returning("rewritten")
        corrective_generate("original query", retriever, critic=critic, max_retries=1)

    # Both scoring calls use the ORIGINAL query, not the reformulated one -
    # scores must be comparable across attempts.
    for call in critic.score_batch_async.await_args_list:
        assert call[0][0] == "original query"


def test_reformulation_failure_falls_back_to_original_query() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("bad")])
    critic = _mock_critic([0.1], Category.INCORRECT)

    with (
        patch("rag_harness.generation.corrective.generate_async", new_callable=AsyncMock),
        patch("rag_harness.generation.corrective.build_async_client") as mock_openai_cls,
    ):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))
        mock_openai_cls.return_value = client
        result = corrective_generate("original", retriever, critic=critic, max_retries=1)

    # Retriever should have been called twice - first with original,
    # then again with original (fallback because reformulation failed).
    assert retriever.retrieve_async.await_count == 2
    for call in retriever.retrieve_async.await_args_list:
        assert call[0][0] == "original"
    assert result.answer == NO_INFO_MESSAGE


def test_reformulation_records_usage_inside_collect_block() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("bad")])
    critic = _mock_critic([0.1], Category.INCORRECT)

    with (
        patch("rag_harness.generation.corrective.generate_async", new_callable=AsyncMock),
        patch("rag_harness.generation.corrective.build_async_client") as mock_openai_cls,
    ):
        mock_openai_cls.return_value = _mock_openai_returning(
            "rewritten", prompt_tokens=15, completion_tokens=5
        )
        with collect_usage() as usage_list:
            corrective_generate("q", retriever, critic=critic, max_retries=1)

    # One reformulation call was made - usage recorded once
    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 15
    assert usage_list[0].output_tokens == 5
