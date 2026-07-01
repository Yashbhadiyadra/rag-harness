"""FastAPI application exposing /query and /health endpoints."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_harness.config import settings
from rag_harness.generation.generator import generate
from rag_harness.logging_setup import configure_logging
from rag_harness.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging once at server startup."""
    configure_logging(settings.log_level)
    yield


app = FastAPI(
    title="RAG Harness",
    description="Reliability-first RAG over Kubernetes documentation.",
    version="0.1.0",
    lifespan=_lifespan,
)

_retriever: DenseRetriever | None = None


def _get_retriever() -> DenseRetriever:
    """Return the module-level retriever, initialising it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = DenseRetriever()
    return _retriever


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    question: str
    top_k: int = settings.retrieval_top_k


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
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and return a grounded answer with source attribution."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    retriever = _get_retriever()
    chunks = retriever.retrieve(request.question, top_k=request.top_k)
    answer = generate(request.question, chunks)

    sources = [Source(source_file=c.source_file, heading_path=c.heading_path) for c in chunks]
    logger.info("query answered — %d sources used", len(sources))
    return QueryResponse(question=request.question, answer=answer, sources=sources)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — returns 200 when the service is running."""
    return {"status": "ok"}
