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
from pathlib import Path
from typing import Any

from rag_harness.config import settings

logger = logging.getLogger(__name__)

_MAX_TOP_K = 50
DEFAULT_EXPERIMENTS_DIR = Path("evals/experiments")


async def query_docs_impl(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer *question* from the pinned corpus with chunk-level citations."""
    from rag_harness.generation.citations import cited_chunk_indices
    from rag_harness.generation.generator import generate_async
    from rag_harness.retrieval.factory import build_retriever

    if not question.strip():
        return {"error": "question must not be empty"}
    top_k = max(1, min(_MAX_TOP_K, top_k))

    retriever = build_retriever(settings.retrieval_strategy)
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
    }


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
    server.add_tool(
        query_docs_impl,
        name="query_docs",
        description=(
            "Answer a question grounded in the pinned documentation corpus, "
            "returning the answer, its source files, and chunk-level citations."
        ),
    )
    server.add_tool(
        get_eval_report_impl,
        name="get_eval_report",
        description=(
            "Latest evaluation metrics for the production retrieval config, "
            "including whether the reliability gate passed."
        ),
    )
    server.add_tool(
        _get_ablation_report_tool,
        name="get_ablation_report",
        description="The latest ablation table comparing retrieval strategies (markdown).",
    )
    return server


def main() -> None:
    """Run the MCP server over stdio (no network surface)."""
    logger.info("starting rag-harness MCP server (stdio transport)")
    build_server().run()
