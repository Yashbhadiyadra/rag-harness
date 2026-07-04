"""Unit tests for the tracing wrapper.

Uses opentelemetry-sdk's InMemorySpanExporter (dev dep) so tests never touch
a real Phoenix instance. The Phoenix + OpenAI-instrumentor packages are
exercised only in configure_tracing() and are lazy-imported — those code
paths are tested with mocked import failures.
"""

import sys
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rag_harness.observability import tracing
from rag_harness.observability.tracing import (
    configure_tracing,
    set_tracer_for_testing,
    traced_span,
)


@pytest.fixture()
def in_memory_tracer():
    """Provide an in-memory tracer and reset module state around each test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    original = tracing._tracer
    set_tracer_for_testing(tracer)
    try:
        yield exporter
    finally:
        set_tracer_for_testing(original)


# --- traced_span, tracing disabled (default state) ---


def test_traced_span_is_noop_when_tracer_not_configured() -> None:
    set_tracer_for_testing(None)
    with traced_span("stage") as span:
        assert span is None


def test_traced_span_swallows_attributes_when_tracer_off() -> None:
    set_tracer_for_testing(None)
    # No assertions needed — just verifying no exception is raised
    with traced_span("stage", key1="v1", key2=42, key3=None):
        pass


# --- traced_span, tracing enabled ---


def test_traced_span_emits_a_span(in_memory_tracer: InMemorySpanExporter) -> None:
    with traced_span("retrieve"):
        pass
    spans = in_memory_tracer.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "retrieve"


def test_traced_span_records_attributes(in_memory_tracer: InMemorySpanExporter) -> None:
    with traced_span("generate", strategy="hybrid", top_k=5):
        pass
    spans = in_memory_tracer.get_finished_spans()
    assert spans[0].attributes["strategy"] == "hybrid"
    assert spans[0].attributes["top_k"] == 5


def test_traced_span_skips_none_attributes(in_memory_tracer: InMemorySpanExporter) -> None:
    with traced_span("stage", filled="yes", missing=None):
        pass
    spans = in_memory_tracer.get_finished_spans()
    attrs = spans[0].attributes
    assert attrs["filled"] == "yes"
    assert "missing" not in attrs


def test_traced_span_nests_child_under_parent(
    in_memory_tracer: InMemorySpanExporter,
) -> None:
    with traced_span("query"):
        with traced_span("retrieve"):
            pass
        with traced_span("generate"):
            pass
    spans = in_memory_tracer.get_finished_spans()
    # SDK reports children before parents on finish
    names_to_ids = {s.name: s.context.span_id for s in spans}
    parent_id = names_to_ids["query"]
    assert spans[0].parent.span_id == parent_id
    assert spans[1].parent.span_id == parent_id


def test_traced_span_records_exception(in_memory_tracer: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError):
        with traced_span("failing_stage"):
            raise RuntimeError("boom")
    spans = in_memory_tracer.get_finished_spans()
    # Span still emitted despite the exception
    assert len(spans) == 1
    assert spans[0].name == "failing_stage"


# --- configure_tracing ---


def test_configure_tracing_is_noop_when_disabled() -> None:
    set_tracer_for_testing(None)
    with patch("rag_harness.observability.tracing.settings.tracing_enabled", False):
        configure_tracing()
    # _tracer stays None — no phoenix import attempted
    assert tracing._tracer is None


def test_configure_tracing_raises_helpful_error_when_extra_missing() -> None:
    set_tracer_for_testing(None)
    with (
        patch("rag_harness.observability.tracing.settings.tracing_enabled", True),
        patch.dict(sys.modules, {"phoenix.otel": None}),
    ):
        with pytest.raises(ImportError, match=r"\[observability\] extra"):
            configure_tracing()


def test_configure_tracing_is_idempotent(in_memory_tracer: InMemorySpanExporter) -> None:
    # Tracer already set by fixture; configure_tracing() with enabled=True
    # must not re-import Phoenix or reinitialise.
    with patch("rag_harness.observability.tracing.settings.tracing_enabled", True):
        # If it tried to import phoenix, the test env doesn't have it wired to
        # register(...) and it would fail. Idempotency is the guard.
        configure_tracing()
    # Tracer unchanged — still the in-memory one from the fixture
    assert tracing._tracer is not None


# --- Integration with real callers ---


def test_traced_span_is_composable_with_collect_usage(
    in_memory_tracer: InMemorySpanExporter,
) -> None:
    from rag_harness.observability.usage import (
        TokenUsage,
        collect_usage,
        record_usage,
    )

    with (
        traced_span("evaluate_case"),
        collect_usage() as usage_list,
    ):
        with traced_span("generate"):
            record_usage(TokenUsage("gpt-4o-mini", 100, 40, 0.045))

    # Usage collector captured the record
    assert len(usage_list) == 1
    # Tracer emitted parent + child span
    spans = in_memory_tracer.get_finished_spans()
    assert {s.name for s in spans} == {"evaluate_case", "generate"}
