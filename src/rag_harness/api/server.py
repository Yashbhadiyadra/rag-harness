"""FastAPI application exposing /query, /health, and /metrics endpoints."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from rag_harness.api.budget import DailyBudget
from rag_harness.api.errors import GuardrailRejection, RagHarnessError
from rag_harness.api.guardrails import screen_for_injection
from rag_harness.api.metrics import (
    QUERY_COST_USD,
    QUERY_ERRORS_TOTAL,
    QUERY_LATENCY_SECONDS,
    QUERY_TOKENS,
    QUERY_TOTAL,
    prometheus_response,
)
from rag_harness.api.middleware.daily_cap import DailyCapMiddleware
from rag_harness.api.middleware.kill_switch import KillSwitchMiddleware
from rag_harness.api.middleware.security_headers import SecurityHeadersMiddleware
from rag_harness.config import settings
from rag_harness.evaluation.closed_loop import capture_query
from rag_harness.generation.citations import cited_chunk_indices
from rag_harness.generation.corrective import NO_INFO_MESSAGE, corrective_generate_async
from rag_harness.generation.generator import generate_async
from rag_harness.logging_setup import configure_logging
from rag_harness.models import Chunk
from rag_harness.observability.tracing import (
    TraceSpan,
    collect_spans,
    configure_tracing,
    traced_span,
)
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


# /docs stays enabled deliberately: the repo is public, so the OpenAPI
# schema reveals nothing an attacker cannot read in source, and the
# interactive docs showcase the API. /redoc is dropped as an unused
# duplicate surface - one documented interface is enough to maintain.
app = FastAPI(
    title="RAG Harness",
    description="Reliability-first RAG over Kubernetes documentation.",
    version="0.1.0",
    lifespan=_lifespan,
    redoc_url=None,
)

# Rate limiting: per-IP by default. See settings.api_rate_limit
# (default 10/hour;3/minute for the public demo - ADR-0010) and .env.example.
# In a single-instance demo the in-memory limiter is fine; swap to a Redis
# backend later without code changes if we horizontally scale.
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.api_rate_limit])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
app.add_middleware(SlowAPIMiddleware)

# Public-demo cost guardrails (ADR-0010). The DailyBudget is shared on
# app.state so tests can reach it via `app.state.daily_budget.reset()`.
#
# Middleware add order matters: the LAST middleware added runs FIRST on
# the inbound path. Order below = security headers → kill switch →
# daily cap → slowapi. Security headers outermost so every response -
# including kill-switch and rate-limit rejections - carries them.
app.state.daily_budget = DailyBudget(cap=settings.demo_daily_request_cap)
app.add_middleware(DailyCapMiddleware, budget=app.state.daily_budget)
app.add_middleware(KillSwitchMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.exception_handler(RagHarnessError)
async def _rag_harness_error_handler(request: Request, exc: RagHarnessError) -> Response:
    """Map every RagHarnessError subclass to a structured JSON response."""
    import json

    logger.warning(
        "api error: %s",
        exc.message,
        extra={
            "error_type": exc.error_type,
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )
    body = {
        "error_type": exc.error_type,
        "message": exc.message,
        "detail": exc.detail,
    }
    return Response(
        content=json.dumps(body),
        status_code=exc.status_code,
        media_type="application/json",
    )


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


class Citation(BaseModel):
    """A passage the answer cited inline, tied to its marker.

    ``marker`` is the 1-based number the answer used (e.g. ``[1]``), so a
    client can link a specific claim to the exact chunk it came from -
    chunk-level attribution, not just the file-level list in ``sources``.
    """

    marker: int
    source_file: str
    heading_path: list[str]


class QueryResponse(BaseModel):
    """Response body returned by POST /query.

    ``trace``, ``cost_usd``, and ``latency_ms`` are populated per-request
    so the demo UI (ADR-0010) can render the per-stage breakdown and the
    this-query cost/latency alongside the answer. API clients that only
    want the answer + sources can ignore the extra fields.
    """

    question: str
    answer: str
    sources: list[Source]
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceSpan] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@app.post("/query", response_model=QueryResponse)
@limiter.limit(settings.api_rate_limit)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and return a grounded answer with source attribution.

    Rate-limited at ``settings.api_rate_limit`` (default 60/minute per IP).
    The ``request`` parameter is required by slowapi even though it is not
    used directly here.

    Wrapped in a ``collect_usage()`` block so every LLM call made downstream
    contributes to the Prometheus ``rag_query_tokens_total`` and
    ``rag_query_cost_usd_total`` counters, and in a ``collect_spans()`` block
    so the per-stage trace can be returned on the response for the demo UI.
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    reason = screen_for_injection(body.question)
    if reason is not None:
        raise GuardrailRejection("input rejected by guardrail", detail=reason)

    retriever = _get_retriever()
    strategy = settings.retrieval_strategy
    use_corrective = (
        body.corrective if body.corrective is not None else settings.corrective_rag_enabled
    )
    corrective_label = "true" if use_corrective else "false"

    start = time.perf_counter()
    chunks: list[Chunk] = []
    answer = ""
    # Collectors outermost so the outer "query" span itself lands in the list.
    with collect_spans() as span_list, collect_usage() as usage_list:
        try:
            with traced_span(
                "query",
                strategy=strategy,
                corrective=use_corrective,
                top_k=body.top_k,
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
            # rather than a 5xx - the API contract stays useful and the user
            # sees the same "not enough information" signal they would from
            # a corrective-flow refusal. Distinct log event for ops.
            QUERY_ERRORS_TOTAL.labels(strategy=strategy, error_type=type(e).__name__).inc()
            logger.warning(
                "LLM boundary exhausted (%s: %s) - returning refusal for query: %.60s",
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

    # Closed-loop eval (ADR-0020): a low-confidence answer signals a golden-set
    # gap. Capture it as a review candidate (never touches the golden files).
    # Opt-in and cheap - the refusal signal needs no extra LLM call.
    if settings.closed_loop_enabled:
        try:
            capture_query(body.question, answer, chunks, Path(settings.closed_loop_queue_path))
        except Exception:  # noqa: BLE001 - capture must never break a response
            logger.warning("closed-loop capture failed", exc_info=True)

    # Aggregate usage from every LLM call into the token and cost counters,
    # and sum for the this-query cost returned on the response.
    cost_usd = 0.0
    for usage in usage_list:
        QUERY_TOKENS.labels(direction="input", model=usage.model).inc(usage.input_tokens)
        QUERY_TOKENS.labels(direction="output", model=usage.model).inc(usage.output_tokens)
        QUERY_COST_USD.inc(usage.estimated_cost_usd)
        cost_usd += usage.estimated_cost_usd

    sources = [Source(source_file=c.source_file, heading_path=c.heading_path) for c in chunks]
    cited_markers = set(cited_chunk_indices(answer))
    citations = [
        Citation(marker=i, source_file=c.source_file, heading_path=c.heading_path)
        for i, c in enumerate(chunks, start=1)
        if i in cited_markers
    ]
    logger.info(
        "query answered - %d sources, %d cited, corrective=%s",
        len(sources),
        len(citations),
        use_corrective,
    )
    return QueryResponse(
        question=body.question,
        answer=answer,
        sources=sources,
        citations=citations,
        trace=span_list,
        cost_usd=cost_usd,
        latency_ms=latency_seconds * 1000.0,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe - returns 200 as long as the process is alive.

    Deliberately trivial: no external checks. Kubernetes-style liveness
    probes should not fail on transient dependency issues (that would
    trigger unnecessary restarts). See ``/ready`` for the dependency check.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Response:
    """Readiness probe - 200 only when every dependency is reachable.

    Checks:
      * ChromaDB heartbeat (round-trip to the persistent store).
      * OPENAI_API_KEY is set (does NOT spend a token verifying it).
      * When RETRIEVAL_STRATEGY requires a cross-encoder, the
        sentence-transformers import succeeds.

    Returns 200 with per-check status on success; 503 with a not_ready
    body listing failed checks on any failure. Failed checks do not
    short-circuit - the whole set runs so an operator can see the
    complete picture in one probe response.
    """
    import json

    checks: dict[str, str] = {}
    failures: list[str] = []

    # ChromaDB
    try:
        import chromadb

        chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        chroma_client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"unreachable: {type(e).__name__}"
        failures.append("chromadb")

    # OpenAI key (never make a real call from a readiness check)
    if settings.openai_api_key and settings.openai_api_key.strip():
        checks["openai_api_key"] = "present"
    else:
        checks["openai_api_key"] = "missing"
        failures.append("openai_api_key")

    # Cross-encoder - only required when the configured strategy uses it
    if settings.retrieval_strategy in ("hybrid-rerank", "full"):
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401

            checks["cross_encoder"] = "importable"
        except ImportError:
            checks["cross_encoder"] = "not-installed"
            failures.append("cross_encoder")

    if failures:
        body_json = {
            "error_type": "not_ready",
            "message": f"not ready: {', '.join(failures)}",
            "detail": None,
            "checks": checks,
        }
        return Response(
            content=json.dumps(body_json),
            status_code=503,
            media_type="application/json",
        )

    return Response(
        content=json.dumps({"status": "ready", "checks": checks}),
        status_code=200,
        media_type="application/json",
    )


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint - RAG-specific counters and histograms."""
    body, content_type = prometheus_response()
    return Response(content=body, media_type=content_type)


# --- Demo UI (ADR-0010 §Demo UI) --------------------------------------
# Static bundle: HTML at /, css/js under /static/. The explicit /
# handler avoids mounting StaticFiles at / (which would shadow route
# path matching). Handlers registered above still resolve first.

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index() -> FileResponse:
    """Serve the demo UI's single HTML page.

    Accepts HEAD as well as GET so uptime monitors (which default to HEAD)
    do not fire false "site down" alerts against the root URL. Starlette
    returns HEAD responses with identical headers and no body.
    """
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")
