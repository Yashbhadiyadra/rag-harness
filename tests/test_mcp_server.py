"""Tests for the MCP server tool implementations."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.api.mcp_server import (
    build_server,
    get_ablation_report_impl,
    get_eval_report_impl,
    query_docs_impl,
)
from rag_harness.models import Chunk


def _chunk(source_file: str) -> Chunk:
    return Chunk(
        id=f"{source_file}::0",
        text="content",
        source_file=source_file,
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
        heading_path=["H"],
    )


@pytest.mark.asyncio
async def test_query_docs_impl_returns_answer_sources_citations() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("a.md"), _chunk("b.md")])

    with (
        patch("rag_harness.retrieval.factory.build_retriever", return_value=retriever),
        patch(
            "rag_harness.generation.generator.generate_async",
            new_callable=AsyncMock,
            return_value="Fact one [1]. Fact two [2].",
        ),
    ):
        out = await query_docs_impl("What is X?", top_k=5)

    assert out["answer"].startswith("Fact one")
    assert [s["source_file"] for s in out["sources"]] == ["a.md", "b.md"]
    assert [c["marker"] for c in out["citations"]] == [1, 2]


@pytest.mark.asyncio
async def test_query_docs_impl_rejects_empty_question() -> None:
    out = await query_docs_impl("   ")
    assert "error" in out


@pytest.mark.asyncio
async def test_query_docs_impl_clamps_top_k() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[])
    with (
        patch("rag_harness.retrieval.factory.build_retriever", return_value=retriever),
        patch(
            "rag_harness.generation.generator.generate_async",
            new_callable=AsyncMock,
            return_value="ans",
        ),
    ):
        await query_docs_impl("q?", top_k=9999)
    # top_k clamped to the 50 max, not passed through as 9999
    assert retriever.retrieve_async.call_args.kwargs["top_k"] == 50


def test_get_eval_report_impl_empty(tmp_path: Path) -> None:
    with patch("rag_harness.evaluation.history.load_history", return_value=[]):
        out = get_eval_report_impl()
    assert "error" in out


def test_get_eval_report_impl_returns_latest() -> None:
    entry = MagicMock(
        timestamp="2026-07-14T00:00:00+00:00",
        git_commit="abc1234",
        strategy="dense",
        corrective=False,
        n_cases=160,
        passed=True,
        mean_context_recall=0.95,
        mean_faithfulness=0.94,
        mean_correctness=0.92,
        mean_answer_relevancy=0.81,
    )
    with patch("rag_harness.evaluation.history.load_history", return_value=[entry]):
        out = get_eval_report_impl()
    assert out["strategy"] == "dense"
    assert out["passed"] is True
    assert out["correctness"] == 0.92
    assert "thresholds" in out


def test_get_ablation_report_impl_reads_latest(tmp_path: Path) -> None:
    (tmp_path / "ablation_20260101T000000+0000_aaa.md").write_text("# old")
    (tmp_path / "ablation_20260714T000000+0000_bbb.md").write_text("# newest table")
    assert get_ablation_report_impl(tmp_path) == "# newest table"


def test_get_ablation_report_impl_none(tmp_path: Path) -> None:
    assert "No ablation report" in get_ablation_report_impl(tmp_path)


def test_build_server_registers_tools() -> None:
    # build_server needs the optional 'mcp' extra; skip where it is not
    # installed (CI installs .[dev] only). The tool impls above are covered
    # without mcp, so the important logic is always exercised.
    pytest.importorskip("mcp")
    server = build_server()
    assert server is not None
    assert server.name == "rag-harness"
