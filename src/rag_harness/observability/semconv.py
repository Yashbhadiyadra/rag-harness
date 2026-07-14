"""OpenTelemetry GenAI semantic-convention attributes.

Maps this project's LLM calls onto the OpenTelemetry GenAI semantic
conventions (https://opentelemetry.io/docs/specs/semconv/gen-ai/) so traces
are portable: any OTEL backend can query them by the same standard attribute
names, not just the Phoenix backend this project happens to configure.

The helper sets the attributes on the *currently active* span, which is the
idiomatic OTEL pattern - the LLM call annotates whatever stage span
(``generate``, ``retrieve``, ...) is active up the call stack. Outside a
span the current span is a non-recording no-op, so callers can invoke this
unconditionally.
"""

from opentelemetry import trace

# GenAI semantic-convention attribute keys (stable subset).
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# gen_ai.system value for the OpenAI-compatible client this project uses.
SYSTEM_OPENAI = "openai"


def genai_attributes(
    operation: str,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, str | int]:
    """Build the GenAI attribute dict for one LLM call.

    ``operation`` is the gen_ai.operation.name (e.g. "chat" for a chat
    completion, "embeddings" for an embedding call). Token counts are
    included only when provided.
    """
    attrs: dict[str, str | int] = {
        GEN_AI_SYSTEM: SYSTEM_OPENAI,
        GEN_AI_OPERATION_NAME: operation,
        GEN_AI_REQUEST_MODEL: model,
    }
    if input_tokens is not None:
        attrs[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    return attrs


def set_current_genai_attributes(
    operation: str,
    model: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Set the GenAI attributes on the currently active span.

    A no-op when no recording span is active (the OTEL current span is then a
    non-recording sentinel whose set_attribute does nothing).
    """
    span = trace.get_current_span()
    for key, value in genai_attributes(operation, model, input_tokens, output_tokens).items():
        span.set_attribute(key, value)
