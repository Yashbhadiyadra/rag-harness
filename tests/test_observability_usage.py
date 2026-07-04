"""Unit tests for the TokenUsage record and collect_usage() context manager."""

from types import SimpleNamespace

import pytest

from rag_harness.observability.usage import (
    TokenUsage,
    _current_usage,
    collect_usage,
    record_usage,
)

# --- TokenUsage.from_openai ---


def test_from_openai_chat_completion() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40))
    usage = TokenUsage.from_openai("gpt-4o-mini", response)
    assert usage.model == "gpt-4o-mini"
    assert usage.input_tokens == 120
    assert usage.output_tokens == 40
    # gpt-4o-mini: 120/1e6 × 0.15 + 40/1e6 × 0.60
    assert usage.estimated_cost_usd == 120 / 1_000_000 * 0.15 + 40 / 1_000_000 * 0.60


def test_from_openai_embedding_has_no_completion_tokens() -> None:
    # Embedding responses omit completion_tokens
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=500))
    usage = TokenUsage.from_openai("text-embedding-3-small", response)
    assert usage.input_tokens == 500
    assert usage.output_tokens == 0
    assert usage.estimated_cost_usd == 500 / 1_000_000 * 0.02


def test_from_openai_missing_usage_field() -> None:
    # Some providers omit `usage` entirely on error paths
    response = SimpleNamespace()
    usage = TokenUsage.from_openai("gpt-4o-mini", response)
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.estimated_cost_usd == 0.0


def test_from_openai_none_values_treated_as_zero() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None))
    usage = TokenUsage.from_openai("gpt-4o-mini", response)
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_token_usage_is_frozen() -> None:
    usage = TokenUsage("gpt-4o-mini", 10, 5, 0.0)
    with pytest.raises((AttributeError, Exception)):
        usage.input_tokens = 999  # type: ignore[misc]


# --- collect_usage() ---


def _make_usage(input_tokens: int = 100) -> TokenUsage:
    return TokenUsage("gpt-4o-mini", input_tokens, 0, 0.0)


def test_collect_usage_captures_records() -> None:
    with collect_usage() as usage_list:
        record_usage(_make_usage(100))
        record_usage(_make_usage(200))
    assert len(usage_list) == 2
    assert usage_list[0].input_tokens == 100
    assert usage_list[1].input_tokens == 200


def test_record_usage_outside_block_is_noop() -> None:
    # No exception, no state leaked
    record_usage(_make_usage(999))
    # After the no-op call, entering a block must give an empty accumulator
    with collect_usage() as usage_list:
        pass
    assert usage_list == []


def test_context_var_is_reset_after_block() -> None:
    assert _current_usage.get() is None
    with collect_usage():
        assert _current_usage.get() is not None
    assert _current_usage.get() is None


def test_context_var_reset_on_exception() -> None:
    with pytest.raises(RuntimeError):
        with collect_usage():
            record_usage(_make_usage(50))
            raise RuntimeError("boom")
    # Exception must not leave the ContextVar set
    assert _current_usage.get() is None


def test_nested_collect_usage_blocks_are_isolated() -> None:
    with collect_usage() as outer:
        record_usage(_make_usage(1))
        with collect_usage() as inner:
            record_usage(_make_usage(2))
        # Inner collected only its own record
        assert len(inner) == 1
        assert inner[0].input_tokens == 2
        # Outer should not see the inner record
        assert len(outer) == 1
        assert outer[0].input_tokens == 1

    # After both blocks exit, further no-op
    record_usage(_make_usage(3))
    assert _current_usage.get() is None
