"""Per-call token usage tracking via a ContextVar collector.

Design decision: usage collection is cross-cutting, so it uses `ContextVar`
rather than threading `(result, TokenUsage)` tuples through every function
signature. Any code path can opt in with a `with collect_usage() as usage:`
block; instrumented call sites call `record_usage(...)` inside the block.
Calls made outside a `collect_usage()` block are silent no-ops — the
collector is opt-in, not always-on. See ADR-0008 for the full rationale.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from rag_harness.observability.pricing import price


@dataclass(frozen=True)
class TokenUsage:
    """Immutable record of one API call's token consumption and estimated cost."""

    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

    @classmethod
    def from_openai(cls, model: str, response: Any) -> "TokenUsage":
        """Extract usage from an OpenAI SDK response object.

        Handles chat completions (`prompt_tokens` + `completion_tokens`) and
        embeddings (`prompt_tokens` only, no completion). Missing usage fields
        default to 0 rather than raising — the observability layer must degrade
        gracefully when a provider omits the field.
        """
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return cls(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=price(model, input_tokens, output_tokens),
        )


_current_usage: ContextVar[list[TokenUsage] | None] = ContextVar("_current_usage", default=None)


@contextmanager
def collect_usage() -> Iterator[list[TokenUsage]]:
    """Collect every `record_usage(...)` call made inside the block into a list.

    Blocks nest safely — inner blocks get their own accumulator and do not
    contaminate the outer one. The ContextVar is reset on both success and
    exception, so a raised exception inside the block does not leak state
    into subsequent calls.
    """
    accumulator: list[TokenUsage] = []
    token = _current_usage.set(accumulator)
    try:
        yield accumulator
    finally:
        _current_usage.reset(token)


def record_usage(usage: TokenUsage) -> None:
    """Append *usage* to the current `collect_usage()` accumulator, if one exists.

    Outside a `collect_usage()` block this is a no-op — instrumented code does
    not need to know whether anyone is listening.
    """
    accumulator = _current_usage.get()
    if accumulator is not None:
        accumulator.append(usage)
