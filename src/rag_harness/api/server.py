"""FastAPI application exposing /query, /health, and /metrics endpoints."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from openai import OpenAIError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_harness.api.metrics import (
    QUERY_COST_USD,
    QUERY_ERRORS_TOTAL,
    QUERY_LATENCY_SECONDS,
    QUERY_TOKENS,
    QUERY_TOTAL,
    prometheus_response,
)
from rag_harness.config import settings
from rag_harness.generation.corrective import NO_INFO_MESSAGE, corrective_generate_async
from rag_harness.generation.generator import generate_async
from rag_harness.logging_setup import configure_logging
from rag_harness.models import Chunk
from rag_harness.observability.tracing import configure_tracing, traced_span
from rag_harness.observability.usage import collect_usage
from rag_harness.retrieval.base import Retriever
from rag_harness.retrieval.factory import build_retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and tracing once at server startup."""
    configure_logging(settings.log_level)
    configure_tracing()
    yield


app = FastAPI(
    title="RAG Harness",
    description="Reliability-first RAG over Kubernetes documentation.",
    version="0.1.0",
    lifespan=_lifespan,
)

# Rate limiting: per-IP by default. See settings.api_rate_limit
# (default 60/minute) and .env.example. In a single-instance demo the
# in-memory limiter is fine; swap to a Redis backend later without code
# changes if we horizontally scale.
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.api_rate_limit])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Return the module-level retriever, initialising it on first call.

    Uses the strategy set by RETRIEVAL_STRATEGY in .env.
    """
    global _retriever
    if _retriever is None:
        _retriever = build_retriever(settings.retrieval_strategy)
    return _retriever


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str = Field(min_length=1, max_length=settings.api_max_question_length)
    top_k: int = Field(default=settings.retrieval_top_k, ge=1, le=50)
    corrective: bool | None = None  # None = fall back to CORRECTIVE_RAG_ENABLED


class Source(BaseModel):
    """A single chunk source attribution included in the query response."""

    source_file: str
    heading_path: list[str]


class QueryResponse(BaseModel):
    """Response body returned by POST /query."""

    question: str
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
@limiter.limit(settings.api_rate_limit)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and return a grounded answer with source attribution.

    Rate-limited at ``settings.api_rate_limit`` (default 60/minute per IP).
    The ``request`` parameter is required by slowapi even though it is not
    used directly here.

    Wrapped in a ``collect_usage()`` block so every LLM call made downstream
    contributes to the Prometheus ``rag_query_tokens_total`` and
    ``rag_query_cost_usd_total`` counters.
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    retriever = _get_retriever()
    strategy = settings.retrieval_strategy
    use_corrective = (
        body.corrective if body.corrective is not None else settings.corrective_rag_enabled
    )
    corrective_label = "true" if use_corrective else "false"

    start = time.perf_counter()
    chunks: list[Chunk] = []
    answer = ""
    try:
        with (
            traced_span(
                "query",
                strategy=strategy,
                corrective=use_corrective,
                top_k=body.top_k,
            ),
            collect_usage() as usage_list,
        ):
            if use_corrective:
                with traced_span("corrective_generate"):
                    result = await corrective_generate_async(
                        body.question, retriever, top_k=body.top_k
                    )
                    answer = result.answer
                    chunks = result.chunks_used
            else:
                with traced_span("retrieve"):
                    chunks = await retriever.retrieve_async(body.question, top_k=body.top_k)
                with traced_span("generate", chunk_count=len(chunks)):
                    answer = await generate_async(body.question, chunks)
    except OpenAIError as e:
        # LLM boundary exhausted its retries. Return the honest refusal
        # rather than a 5xx — the API contract stays useful and the user
        # sees the same "not enough information" signal they would from
        # a corrective-flow refusal. Distinct log event for ops.
        QUERY_ERRORS_TOTAL.labels(strategy=strategy, error_type=type(e).__name__).inc()
        logger.warning(
            "LLM boundary exhausted (%s: %s) — returning refusal for query: %.60s",
            type(e).__name__,
            e,
            body.question,
        )
        answer = NO_INFO_MESSAGE
        chunks = []
    except Exception as e:
        QUERY_ERRORS_TOTAL.labels(strategy=strategy, error_type=type(e).__name__).inc()
        raise
    finally:
        latency_seconds = time.perf_counter() - start
        QUERY_LATENCY_SECONDS.labels(strategy=strategy).observe(latency_seconds)

    QUERY_TOTAL.labels(strategy=strategy, corrective=corrective_label).inc()

    # Aggregate usage from every LLM call into the token and cost counters.
    for usage in usage_list:
        QUERY_TOKENS.labels(direction="input", model=usage.model).inc(usage.input_tokens)
        QUERY_TOKENS.labels(direction="output", model=usage.model).inc(usage.output_tokens)
        QUERY_COST_USD.inc(usage.estimated_cost_usd)

    sources = [Source(source_file=c.source_file, heading_path=c.heading_path) for c in chunks]
    logger.info("query answered — %d sources used, corrective=%s", len(sources), use_corrective)
    return QueryResponse(question=body.question, answer=answer, sources=sources)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — returns 200 when the service is running."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint — RAG-specific counters and histograms."""
    body, content_type = prometheus_response()
    return Response(content=body, media_type=content_type)
