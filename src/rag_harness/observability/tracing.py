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

Turn it on with ``TRACING_ENABLED=true`` in ``.env`` and run Phoenix locally
(``docker compose up phoenix``) or self-host it wherever the deploy lives.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Tracer

from rag_harness.config import settings

logger = logging.getLogger(__name__)

_tracer: Tracer | None = None
_INSTALL_HINT = (
    "Tracing requires the [observability] extra. Install with: pip install -e '.[observability]'"
)


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
    """Context manager for a stage span. Silent no-op when tracing is off.

    Attributes with ``None`` values are skipped so callers can pass optional
    labels unconditionally without polluting spans with null values.
    """
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is None:
                continue
            span.set_attribute(key, value)
        yield span
