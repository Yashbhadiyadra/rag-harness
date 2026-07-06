"""OpenTelemetry per-stage tracing with Arize Phoenix as the backend.

Design (per ADR-0009):

- Uses the OpenTelemetry API (in core deps) so the ``traced_span`` context
  manager is safe to call from anywhere. When tracing is disabled it is a
  cheap no-op.
- Phoenix-specific and OpenAI-instrumentation packages are lazy-imported inside
  ``configure_tracing()`` and live behind the ``[observability]`` extra. The
  API-only import keeps the base install lightweight.
- Instrumented at the RAG stage boundaries (``retrieve``, ``generate``, ``score``)
  in the eval runner and the API server. OpenAI SDK calls made inside each
  stage nest automatically thanks to ``openinference-instrumentation-openai``.
- ``collect_spans()`` is an in-request span collector that runs independently
  of Phoenix. It lets the API return the trace on the response payload so the
  demo UI can render it (ADR-0010) even when Phoenix is not reachable — which
  is the default on Cloud Run.

Turn Phoenix export on with ``TRACING_ENABLED=true`` in ``.env`` and run
Phoenix locally (``docker compose up phoenix``) or self-host it. The
in-request collector is always available regardless.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer
from pydantic import BaseModel, Field

from rag_harness.config import settings

logger = logging.getLogger(__name__)

_tracer: Tracer | None = None
_INSTALL_HINT = (
    "Tracing requires the [observability] extra. Install with: pip install -e '.[observability]'"
)


class TraceSpan(BaseModel):
    """One stage of a query trace.

    Doubles as both the collector's internal type and the API response
    schema — pydantic serialises it directly on ``QueryResponse.trace``.
    """

    name: str
    duration_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)


_current_spans: ContextVar[list[TraceSpan] | None] = ContextVar("_current_spans", default=None)


@contextmanager
def collect_spans() -> Iterator[list[TraceSpan]]:
    """Collect every ``traced_span`` that completes inside the block.

    Blocks nest safely — inner blocks get their own accumulator and do not
    contaminate the outer one. The ContextVar is reset on both success and
    exception, so a raised exception inside the block does not leak state
    into subsequent calls.
    """
    accumulator: list[TraceSpan] = []
    token = _current_spans.set(accumulator)
    try:
        yield accumulator
    finally:
        _current_spans.reset(token)


def _record_span(
    accumulator: list[TraceSpan] | None,
    name: str,
    start: float,
    filtered_attrs: dict[str, Any],
) -> None:
    """Append a TraceSpan to the active collector, if any. No-op otherwise."""
    if accumulator is None:
        return
    duration_ms = (time.perf_counter() - start) * 1000.0
    accumulator.append(TraceSpan(name=name, duration_ms=duration_ms, attributes=filtered_attrs))


def configure_tracing() -> None:
    """Initialise the Phoenix tracer if TRACING_ENABLED is set. Idempotent.

    Called from the FastAPI lifespan hook and from the CLI entry point. If
    tracing is disabled this is a cheap no-op — no imports, no network.
    Raises ImportError with an install hint if tracing is enabled but the
    ``[observability]`` extra is not installed.
    """
    global _tracer
    if not settings.tracing_enabled:
        logger.debug("tracing disabled")
        return
    if _tracer is not None:
        logger.debug("tracing already configured — skipping re-init")
        return

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    tracer_provider = register(
        endpoint=settings.tracing_endpoint,
        project_name=settings.tracing_service_name,
    )
    # Auto-instrument OpenAI SDK calls so they nest under our stage spans
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    _tracer = trace.get_tracer("rag_harness")
    logger.info(
        "tracing configured — exporting to %s (project=%s)",
        settings.tracing_endpoint,
        settings.tracing_service_name,
    )


def set_tracer_for_testing(tracer: Tracer | None) -> None:
    """Inject a tracer for tests (e.g. one backed by InMemorySpanExporter).

    Not intended for production use. Kept public for testability rather than
    private-name-mangling tests.
    """
    global _tracer
    _tracer = tracer


@contextmanager
def traced_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Context manager for a stage span.

    Attributes with ``None`` values are skipped so callers can pass optional
    labels unconditionally without polluting spans with null values.

    Independently of the Phoenix tracer, if a ``collect_spans()`` block is
    active up the call stack the span's name / duration / attributes are
    appended to its accumulator when the span closes. This runs whether or
    not ``TRACING_ENABLED`` is set, so the API can return the trace on the
    response payload for the demo UI.
    """
    start = time.perf_counter()
    accumulator = _current_spans.get()
    filtered_attrs = {k: v for k, v in attributes.items() if v is not None}

    if _tracer is None:
        try:
            yield None
        finally:
            _record_span(accumulator, name, start, filtered_attrs)
        return

    with _tracer.start_as_current_span(name) as span:
        for key, value in filtered_attrs.items():
            span.set_attribute(key, value)
        try:
            yield span
        finally:
            _record_span(accumulator, name, start, filtered_attrs)
