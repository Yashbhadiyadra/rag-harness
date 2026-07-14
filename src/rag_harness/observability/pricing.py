"""Model pricing table and per-call cost calculation.

Rates are USD per **1 million tokens**, quoted from the model provider's public
price sheet as of the last CHANGELOG entry. Embedding models have no output
tokens; the output rate is 0.0. Users can override any rate without editing this
file via `MODEL_RATES_OVERRIDES` in `.env`.
"""

from rag_harness.config import settings

# (input_rate_per_million, output_rate_per_million) in USD.
MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
}


def _resolve_rates(model: str) -> tuple[float, float]:
    """Return (input_rate, output_rate) for *model*, honoring config overrides."""
    if model in settings.model_rates_overrides:
        override = settings.model_rates_overrides[model]
        return (override[0], override[1])
    return MODEL_RATES.get(model, (0.0, 0.0))


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost of a single API call.

    Returns 0.0 for unknown models rather than raising - an unpriced call should
    not break the observability layer.
    """
    input_rate, output_rate = _resolve_rates(model)
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
