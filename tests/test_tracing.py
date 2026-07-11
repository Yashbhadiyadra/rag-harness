"""Unit tests for the tracing wrapper.

Uses opentelemetry-sdk's InMemorySpanExporter (dev dep) so tests never touch
a real Phoenix instance. The Phoenix + OpenAI-instrumentor packages are
exercised only in configure_tracing() and are lazy-imported - those code
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
    collect_spans,
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
    # No assertions needed - just verifying no exception is raised
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
    # _tracer stays None - no phoenix import attempted
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
    # Tracer unchanged - still the in-memory one from the fixture
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


# --- collect_spans (in-request collector) ---


def test_collect_spans_records_completed_spans_when_tracer_off() -> None:
    """Collector must work without the Phoenix tracer - that is its purpose.

    The API returns the trace on the response even when TRACING_ENABLED is
    unset (the Cloud Run default), so the collector cannot depend on the
    tracer being configured.
    """
    set_tracer_for_testing(None)
    with collect_spans() as spans:
        with traced_span("retrieve", top_k=3):
            pass
        with traced_span("generate", chunk_count=5):
            pass

    assert [s.name for s in spans] == ["retrieve", "generate"]
    assert spans[0].attributes == {"top_k": 3}
    assert spans[1].attributes == {"chunk_count": 5}
    assert all(s.duration_ms >= 0 for s in spans)


def test_collect_spans_captures_nested_span_order() -> None:
    """Children close before parents, so a nested trace lists children first."""
    set_tracer_for_testing(None)
    with collect_spans() as spans:
        with traced_span("query"):
            with traced_span("retrieve"):
                pass
            with traced_span("generate"):
                pass

    assert [s.name for s in spans] == ["retrieve", "generate", "query"]


def test_traced_span_outside_collect_spans_does_not_record() -> None:
    """When no collector is active, traced_span must not leak state anywhere."""
    set_tracer_for_testing(None)
    # Just running traced_span without a collector is a no-op regarding state
    with traced_span("stray"):
        pass
    # And a subsequent collect_spans block sees only what happens inside it
    with collect_spans() as spans:
        with traced_span("real"):
            pass
    assert [s.name for s in spans] == ["real"]


def test_collect_spans_records_span_even_when_exception_raised() -> None:
    set_tracer_for_testing(None)
    with pytest.raises(RuntimeError):
        with collect_spans() as spans:
            with traced_span("failing"):
                raise RuntimeError("boom")
    assert [s.name for s in spans] == ["failing"]


def test_collect_spans_nests_safely() -> None:
    """Inner collect_spans must not contaminate the outer collector."""
    set_tracer_for_testing(None)
    with collect_spans() as outer:
        with traced_span("outer_stage"):
            with collect_spans() as inner:
                with traced_span("inner_stage"):
                    pass
        with traced_span("after_inner"):
            pass

    assert [s.name for s in inner] == ["inner_stage"]
    # Outer sees its own stages but NOT the inner collector's span
    outer_names = [s.name for s in outer]
    assert "inner_stage" not in outer_names
    assert set(outer_names) == {"outer_stage", "after_inner"}
