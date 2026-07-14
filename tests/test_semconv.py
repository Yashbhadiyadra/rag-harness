"""Tests for the OpenTelemetry GenAI semantic-convention helpers."""

from unittest.mock import MagicMock, patch

from rag_harness.observability import semconv


def test_genai_attributes_minimal() -> None:
    attrs = semconv.genai_attributes("chat", "gpt-4o-mini")
    assert attrs == {
        "gen_ai.system": "openai",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "gpt-4o-mini",
    }


def test_genai_attributes_with_usage() -> None:
    attrs = semconv.genai_attributes("chat", "gpt-4o-mini", input_tokens=120, output_tokens=30)
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert attrs["gen_ai.usage.output_tokens"] == 30


def test_genai_attributes_omits_none_tokens() -> None:
    attrs = semconv.genai_attributes("embeddings", "text-embedding-3-small")
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.usage.output_tokens" not in attrs
    assert attrs["gen_ai.operation.name"] == "embeddings"


def test_set_current_genai_attributes_sets_on_active_span() -> None:
    span = MagicMock()
    with patch("rag_harness.observability.semconv.trace.get_current_span", return_value=span):
        semconv.set_current_genai_attributes("chat", "gpt-4o-mini", 100, 20)

    keys_set = {call.args[0] for call in span.set_attribute.call_args_list}
    assert "gen_ai.system" in keys_set
    assert "gen_ai.request.model" in keys_set
    assert "gen_ai.usage.input_tokens" in keys_set
    assert "gen_ai.usage.output_tokens" in keys_set
    # exact value round-trips
    values = {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}
    assert values["gen_ai.request.model"] == "gpt-4o-mini"
    assert values["gen_ai.usage.input_tokens"] == 100
