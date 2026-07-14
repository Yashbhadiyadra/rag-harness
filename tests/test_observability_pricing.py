"""Unit tests for the model pricing table."""

from unittest.mock import patch

from rag_harness.observability.pricing import MODEL_RATES, price


def test_price_input_only_zero_output() -> None:
    # 1,000,000 input tokens × $0.02 = $0.02 (text-embedding-3-small has output rate 0)
    assert price("text-embedding-3-small", 1_000_000, 0) == 0.02


def test_price_input_and_output_are_summed() -> None:
    # gpt-4o-mini: 1M input × $0.15 + 1M output × $0.60 = $0.75
    assert price("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_price_zero_tokens_is_zero() -> None:
    assert price("gpt-4o-mini", 0, 0) == 0.0


def test_price_gpt_4o_is_priced() -> None:
    # gpt-4o: 1M input × $2.50 + 1M output × $10.00 = $12.50 (used by the
    # judge selection matrix - an unpriced judge would report a false $0).
    assert price("gpt-4o", 1_000_000, 1_000_000) == 12.50
    assert "gpt-4o" in MODEL_RATES


def test_price_fractional_tokens() -> None:
    # 150 input tokens on gpt-4o-mini = 150 / 1e6 × 0.15 = 2.25e-5
    result = price("gpt-4o-mini", 150, 0)
    assert result == 150 / 1_000_000 * 0.15


def test_price_unknown_model_returns_zero() -> None:
    # Unknown model must not raise - observability layer degrades gracefully
    assert price("some-model-that-does-not-exist", 1_000_000, 1_000_000) == 0.0


def test_price_override_wins_over_builtin() -> None:
    override = {"gpt-4o-mini": (1.0, 2.0)}  # 1M input × $1 + 1M output × $2 = $3
    with patch("rag_harness.observability.pricing.settings.model_rates_overrides", override):
        assert price("gpt-4o-mini", 1_000_000, 1_000_000) == 3.0


def test_price_override_only_applies_to_named_model() -> None:
    override = {"custom-model": (5.0, 10.0)}
    with patch("rag_harness.observability.pricing.settings.model_rates_overrides", override):
        # gpt-4o-mini still uses builtin rates
        assert price("gpt-4o-mini", 1_000_000, 0) == 0.15
        # custom-model uses the override
        assert price("custom-model", 1_000_000, 0) == 5.0


def test_builtin_rates_include_expected_models() -> None:
    # Guard against accidental deletion of rates for models we ship with
    assert "gpt-4o-mini" in MODEL_RATES
    assert "text-embedding-3-small" in MODEL_RATES
    # Embedding model must have output rate 0
    assert MODEL_RATES["text-embedding-3-small"][1] == 0.0
