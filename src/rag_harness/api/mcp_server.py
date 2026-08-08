"""MCP server exposing the harness as agent tools (ADR-0021).

Model Context Protocol lets an agent (Claude Desktop, Cursor, ...) call the
harness directly. This server exposes a small, read-mostly tool set.

Secure by default:
- **stdio transport only.** The server talks over stdin/stdout to the local
  MCP client that launched it. There is no network listener, so there is
  nothing to expose to the internet - the strongest form of "not exposed
  without auth". Given the 2026 MCP CVE record (thousands of unauthenticated
  internet-exposed servers), refusing to open a network port by default is
  the single most important control.
- **Read-mostly tools.** query_docs answers from the pinned corpus;
  get_eval_report and get_ablation_report return existing measured results.
  No tool mutates state, runs a shell, writes files, or triggers an expensive
  job. There is no destructive action to authorise.
- **Bounded input.** top_k is clamped to the same range as the HTTP API.

An HTTP transport would need the CSA baseline (OAuth 2.1 + PKCE, tool-level
scopes, short token lifetimes, default-deny) before exposure; it is
deliberately not enabled here.

Optional dependency: install with ``pip install -e '.[mcp]'``. The mcp SDK is
lazy-imported so the base package does not require it.
"""

import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from rag_harness.api.auth import TenantContext
from rag_harness.config import settings
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_MAX_TOP_K = 50
DEFAULT_EXPERIMENTS_DIR = Path("evals/experiments")

# Tool descriptions, shared by the stdio and HTTP servers so the two expose an
# identical tool surface.
_QUERY_DESC = (
    "Answer a question grounded in the pinned documentation corpus, returning "
    "the answer, its source files, chunk-level citations, and corpus provenance."
)
_EVAL_DESC = (
    "Latest evaluation metrics for the production retrieval config, including "
    "whether the reliability gate passed."
)
_ABLATION_DESC = "The latest ablation table comparing retrieval strategies (markdown)."

# The authenticated tenant for the current HTTP request. The auth middleware
# sets it from the bearer token; the query tool reads it to select the tenant's
# collection. None means single-tenant / stdio (the default collection).
_current_tenant: ContextVar[TenantContext | None] = ContextVar("_current_tenant", default=None)


def _provenance(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Corpus provenance behind an answer: which pinned snapshot produced it.

    Surfacing the corpus commit and doc version in the response is the
    payload-trust move (ADR-0028): a caller can verify the exact source snapshot,
    not just trust that an answer is grounded.
    """
    return {
        "corpus_commits": sorted({c.git_commit for c in chunks}),
        "doc_versions": sorted({c.doc_version for c in chunks}),
    }


async def _answer(question: str, top_k: int, collection: str | None) -> dict[str, Any]:
    """Retrieve, generate, and shape the grounded answer for one question.

    *collection* selects the tenant's corpus (None = default collection). Shared
    by the stdio tool (default collection) and the HTTP tool (tenant-scoped).
    """
    from rag_harness.generation.citations import cited_chunk_indices
    from rag_harness.generation.generator import generate_async
    from rag_harness.retrieval.factory import build_retriever

    if not question.strip():
        return {"error": "question must not be empty"}
    top_k = max(1, min(_MAX_TOP_K, top_k))

    retriever = build_retriever(settings.retrieval_strategy, collection_name=collection)
    chunks = await retriever.retrieve_async(question, top_k=top_k)
    answer = await generate_async(question, chunks)
    cited = set(cited_chunk_indices(answer))
    return {
        "answer": answer,
        "sources": [{"source_file": c.source_file, "heading_path": c.heading_path} for c in chunks],
        "citations": [
            {"marker": i, "source_file": c.source_file}
            for i, c in enumerate(chunks, start=1)
            if i in cited
        ],
        "provenance": _provenance(chunks),
    }


async def query_docs_impl(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer *question* from the default corpus with chunk-level citations (stdio)."""
    return await _answer(question, top_k, collection=None)


async def query_docs_http_impl(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer *question* scoped to the authenticated tenant's corpus (HTTP).

    The collection is derived from the request's resolved tenant, never from a
    client argument, so a caller cannot retrieve a corpus it was not issued.
    """
    tenant = _current_tenant.get()
    collection = tenant.collection if tenant is not None else None
    return await _answer(question, top_k, collection=collection)


def get_eval_report_impl() -> dict[str, Any]:
    """Latest evaluation metrics from the eval history, with the gate result."""
    from rag_harness.evaluation.history import load_history

    entries = load_history()
    if not entries:
        return {"error": "no eval history yet - run the eval or ablation first"}
    latest = entries[-1]
    return {
        "timestamp": latest.timestamp,
        "commit": latest.git_commit,
        "strategy": latest.strategy,
        "corrective": latest.corrective,
        "n_cases": latest.n_cases,
        "passed": latest.passed,
        "context_recall": round(latest.mean_context_recall, 3),
        "faithfulness": round(latest.mean_faithfulness, 3),
        "correctness": round(latest.mean_correctness, 3),
        "answer_relevancy": round(latest.mean_answer_relevancy, 3),
        "thresholds": {
            "context_recall": settings.threshold_context_recall,
            "faithfulness": settings.threshold_faithfulness,
            "correctness": settings.threshold_correctness,
        },
    }


def get_ablation_report_impl(experiments_dir: Path | None = None) -> str:
    """Return the latest ablation comparison table (markdown), or a note."""
    experiments_dir = experiments_dir or DEFAULT_EXPERIMENTS_DIR
    files = sorted(experiments_dir.glob("ablation_*.md"))
    if not files:
        return "No ablation report yet. Run `rag_harness ablation` to produce one."
    return files[-1].read_text()


def _get_ablation_report_tool() -> str:
    """The latest ablation table comparing retrieval strategies (markdown)."""
    return get_ablation_report_impl()


def build_server() -> Any:
    """Construct the FastMCP server with the tools registered.

    Raises ImportError with install guidance if the mcp extra is absent.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("the MCP server needs the 'mcp' extra: pip install -e '.[mcp]'") from e

    server = FastMCP("rag-harness")
    # add_tool (a method call, not a decorator) registers the typed impls
    # directly, which keeps mypy strict happy even when the mcp SDK is absent
    # and FastMCP resolves to Any.
    server.add_tool(query_docs_impl, name="query_docs", description=_QUERY_DESC)
    server.add_tool(get_eval_report_impl, name="get_eval_report", description=_EVAL_DESC)
    server.add_tool(
        _get_ablation_report_tool, name="get_ablation_report", description=_ABLATION_DESC
    )
    return server


class TenantAuthMiddleware:
    """ASGI middleware resolving the bearer token to a tenant (ADR-0028).

    When ``API_AUTH_ENABLED`` is on, a missing or unknown key is rejected with
    401 and ``WWW-Authenticate: Bearer`` before the request reaches any tool.
    When off (the public demo), the request resolves to the default tenant. The
    resolved tenant is stored in ``_current_tenant`` for the request duration so
    the query tool retrieves only that tenant's corpus.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        from rag_harness.api.auth import _extract_bearer, default_tenant, resolve_tenant

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        token = _extract_bearer(headers.get("authorization"))

        if settings.api_auth_enabled:
            tenant = resolve_tenant(token) if token is not None else None
            if tenant is None:
                await self._reject(send)
                return
        else:
            tenant = default_tenant()

        ctx_token = _current_tenant.set(tenant)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_tenant.reset(ctx_token)

    @staticmethod
    async def _reject(send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"www-authenticate", b"Bearer"),
                    (b"content-type", b"application/json"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b'{"error":"valid API key required"}'})


def build_http_app() -> Any:
    """Construct the stateless streamable-HTTP ASGI app, auth-wrapped (ADR-0028).

    Raises ImportError with install guidance if the mcp extra is absent.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("the MCP server needs the 'mcp' extra: pip install -e '.[mcp]'") from e

    server = FastMCP("rag-harness", stateless_http=True)
    server.add_tool(query_docs_http_impl, name="query_docs", description=_QUERY_DESC)
    server.add_tool(get_eval_report_impl, name="get_eval_report", description=_EVAL_DESC)
    server.add_tool(
        _get_ablation_report_tool, name="get_ablation_report", description=_ABLATION_DESC
    )
    return TenantAuthMiddleware(server.streamable_http_app())


def main() -> None:
    """Run the MCP server over stdio (no network surface)."""
    logger.info("starting rag-harness MCP server (stdio transport)")
    build_server().run()


def main_http(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the stateless HTTP MCP server (ADR-0028)."""
    import uvicorn

    logger.info("starting rag-harness MCP server (stateless HTTP) on %s:%d", host, port)
    uvicorn.run(build_http_app(), host=host, port=port)
