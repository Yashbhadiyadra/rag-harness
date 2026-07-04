"""Unit tests for the RelevanceCritic."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.generation.critic import Category, RelevanceCritic
from rag_harness.models import Chunk
from rag_harness.observability.usage import collect_usage


def _chunk(cid: str, text: str = "") -> Chunk:
    return Chunk(
        id=cid,
        text=text or f"text for {cid}",
        source_file=f"docs/{cid}.md",
        git_commit="abc123",
        doc_version="v1.32",
        chunk_index=0,
        heading_path=[],
    )


def _openai_returning_json(payload: dict) -> MagicMock:
    """Return an AsyncOpenAI client whose async .create returns *payload*."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# --- score_batch ---


def test_score_batch_parses_structured_response() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = _openai_returning_json({"scores": [0.9, 0.5, 0.1]})
        critic = RelevanceCritic()
        scores = critic.score_batch("query", chunks)
    assert scores == [0.9, 0.5, 0.1]


def test_score_batch_empty_input_returns_empty() -> None:
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic()
        assert critic.score_batch("query", []) == []


def test_score_batch_clamps_scores_to_unit_interval() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = _openai_returning_json({"scores": [1.5, -0.2]})
        critic = RelevanceCritic()
        scores = critic.score_batch("query", chunks)
    assert scores == [1.0, 0.0]


def test_score_batch_pads_when_llm_returns_too_few_scores() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = _openai_returning_json({"scores": [0.8]})
        critic = RelevanceCritic()
        scores = critic.score_batch("query", chunks)
    # Padded with zeros so caller can safely zip
    assert scores == [0.8, 0.0, 0.0]


def test_score_batch_truncates_when_llm_returns_too_many_scores() -> None:
    chunks = [_chunk("a")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = _openai_returning_json({"scores": [0.9, 0.4, 0.1]})
        critic = RelevanceCritic()
        scores = critic.score_batch("query", chunks)
    assert scores == [0.9]


def test_score_batch_falls_back_to_zeros_on_openai_error() -> None:
    chunks = [_chunk("a"), _chunk("b")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("api down")
        mock_openai.return_value = client
        critic = RelevanceCritic()
        scores = critic.score_batch("query", chunks)
    assert scores == [0.0, 0.0]


def test_score_batch_falls_back_to_zeros_on_invalid_json() -> None:
    chunks = [_chunk("a")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "not valid json"
        client.chat.completions.create.return_value = resp
        mock_openai.return_value = client
        critic = RelevanceCritic()
        scores = critic.score_batch("query", chunks)
    assert scores == [0.0]


# --- categorise ---


def test_categorise_high_top_score_is_correct() -> None:
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic(correct_threshold=0.7, incorrect_threshold=0.3)
        assert critic.categorise([0.9, 0.2, 0.1]) is Category.CORRECT


def test_categorise_medium_top_score_is_ambiguous() -> None:
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic(correct_threshold=0.7, incorrect_threshold=0.3)
        assert critic.categorise([0.5, 0.4, 0.3]) is Category.AMBIGUOUS


def test_categorise_low_top_score_is_incorrect() -> None:
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic(correct_threshold=0.7, incorrect_threshold=0.3)
        assert critic.categorise([0.2, 0.1, 0.0]) is Category.INCORRECT


def test_categorise_empty_scores_is_incorrect() -> None:
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic()
        assert critic.categorise([]) is Category.INCORRECT


def test_categorise_at_exact_correct_threshold_is_correct() -> None:
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic(correct_threshold=0.7, incorrect_threshold=0.3)
        assert critic.categorise([0.7]) is Category.CORRECT


def test_categorise_at_exact_incorrect_threshold_is_ambiguous() -> None:
    # 0.3 is the boundary — anything ≥ 0.3 and < 0.7 is Ambiguous
    with patch("rag_harness.generation.critic.AsyncOpenAI"):
        critic = RelevanceCritic(correct_threshold=0.7, incorrect_threshold=0.3)
        assert critic.categorise([0.3]) is Category.AMBIGUOUS


# --- usage recording ---


def test_score_batch_records_usage_inside_collect_block() -> None:
    chunks = [_chunk("a")]
    with patch("rag_harness.generation.critic.AsyncOpenAI") as mock_openai:
        client = MagicMock()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps({"scores": [0.9]})
        resp.usage = MagicMock(prompt_tokens=50, completion_tokens=8)
        client.chat.completions.create = AsyncMock(return_value=resp)
        mock_openai.return_value = client

        with collect_usage() as usage_list:
            critic = RelevanceCritic()
            critic.score_batch("query", chunks)

    assert len(usage_list) == 1
    assert usage_list[0].input_tokens == 50
    assert usage_list[0].output_tokens == 8
