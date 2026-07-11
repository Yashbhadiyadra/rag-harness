"""Prometheus metrics for the API server.

Exposes RAG-specific counters, histograms, and gauges via ``GET /metrics``.
Uses the standard prometheus_client library - labels, histogram buckets, and
text-format exposition are provided by the library rather than hand-rolled.

Default process/GC metrics are disabled to keep the /metrics output focused on
RAG behaviour. If cluster ops later want CPU/memory metrics they can add the
default collectors back in a deployment-specific wrapper.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

# Disable the default collectors (process, platform, GC) - we want /metrics
# focused on RAG behaviour. Unregister any that got auto-registered on import.
for collector in list(REGISTRY._collector_to_names.keys()):
    if isinstance(collector, ProcessCollector | PlatformCollector | GCCollector):
        REGISTRY.unregister(collector)


# --- Query-level metrics ---

QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total number of /query requests processed.",
    labelnames=("strategy", "corrective"),
)

QUERY_ERRORS_TOTAL = Counter(
    "rag_query_errors_total",
    "Total number of /query requests that raised an error.",
    labelnames=("strategy", "error_type"),
)

QUERY_LATENCY_SECONDS = Histogram(
    "rag_query_latency_seconds",
    "End-to-end latency of a /query request from receipt to response.",
    labelnames=("strategy",),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# --- Token and cost accounting ---

QUERY_TOKENS = Counter(
    "rag_query_tokens_total",
    "Cumulative input and output tokens consumed by /query requests.",
    labelnames=("direction", "model"),
)

QUERY_COST_USD = Counter(
    "rag_query_cost_usd_total",
    "Cumulative estimated USD cost of /query requests.",
)


def prometheus_response() -> tuple[bytes, str]:
    """Return (body, content_type) for a /metrics endpoint response."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
