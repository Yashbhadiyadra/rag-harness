"""Unit tests for the ablation runner and its markdown/CSV renderers.

All LLM calls and retrievers are mocked - nothing here hits the network.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rag_harness.evaluation.ablation import (
    AblationRun,
    is_relevant_but_incorrect,
    relevant_but_incorrect_cases,
    render_csv,
    render_markdown,
    run_ablation,
)
from rag_harness.models import EvalResult, EvalSummary


def _make_result(
    case_id: str,
    relevancy: float,
    correctness: float,
    context_recall: float = 1.0,
    context_precision: float = 0.9,
    faithfulness: float = 0.9,
    latency_ms: float = 200.0,
    cost: float = 0.001,
) -> EvalResult:
    return EvalResult(
        case_id=case_id,
        question="Q?",
        generated_answer="A.",
        retrieved_doc_ids=["docs/a.md"],
        context_recall=context_recall,
        context_precision=context_precision,
        faithfulness=faithfulness,
        correctness=correctness,
        answer_relevancy=relevancy,
        latency_ms=latency_ms,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=cost,
    )


def _make_summary(results: list[EvalResult], passed: bool = True) -> EvalSummary:
    n = len(results) or 1
    return EvalSummary(
        results=results,
        mean_context_recall=sum(r.context_recall for r in results) / n,
        mean_context_precision=sum(r.context_precision for r in results) / n,
        mean_faithfulness=sum(r.faithfulness for r in results) / n,
        mean_correctness=sum(r.correctness for r in results) / n,
        mean_answer_relevancy=sum(r.answer_relevancy for r in results) / n,
        latency_p50_ms=100.0,
        latency_p95_ms=250.0,
        total_cost_usd=sum(r.estimated_cost_usd for r in results),
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        passed=passed,
    )


# --- is_relevant_but_incorrect ---


def test_rbi_high_relevancy_low_correctness_is_flagged() -> None:
    r = _make_result("c1", relevancy=0.9, correctness=0.2)
    # Default thresholds: relevancy>0.7 AND correctness<0.5
    assert is_relevant_but_incorrect(r) is True


def test_rbi_high_relevancy_high_correctness_is_not_flagged() -> None:
    r = _make_result("c1", relevancy=0.9, correctness=0.9)
    assert is_relevant_but_incorrect(r) is False


def test_rbi_low_relevancy_low_correctness_is_not_flagged() -> None:
    # Off-topic AND wrong is a different failure - not the RBI category
    r = _make_result("c1", relevancy=0.1, correctness=0.0)
    assert is_relevant_but_incorrect(r) is False


def test_rbi_custom_thresholds_respected() -> None:
    r = _make_result("c1", relevancy=0.6, correctness=0.3)
    # Default (0.7/0.5): relevancy 0.6 fails → not RBI
    assert is_relevant_but_incorrect(r) is False
    # Lower threshold to 0.5: now flagged
    assert is_relevant_but_incorrect(r, relevancy_min=0.5, correctness_max=0.5) is True


def test_relevant_but_incorrect_cases_filters_summary() -> None:
    results = [
        _make_result("hit", relevancy=0.9, correctness=0.2),  # RBI
        _make_result("clean", relevancy=0.9, correctness=0.9),  # correct
        _make_result("offtopic", relevancy=0.1, correctness=0.0),  # off-topic
        _make_result("borderline", relevancy=0.9, correctness=0.5),  # correct not < 0.5
    ]
    summary = _make_summary(results)
    rbi = relevant_but_incorrect_cases(summary)
    assert [r.case_id for r in rbi] == ["hit"]


# --- run_ablation ---


def test_run_ablation_invokes_run_eval_per_configuration() -> None:
    dummy_summary = _make_summary([_make_result("c1", relevancy=0.9, correctness=0.9)])

    with (
        patch(
            "rag_harness.evaluation.ablation.build_retriever",
            return_value=MagicMock(),
        ) as mock_build,
        patch(
            "rag_harness.evaluation.ablation.run_eval",
            new_callable=AsyncMock,
            return_value=dummy_summary,
        ) as mock_run_eval,
    ):
        runs = asyncio.run(run_ablation())

    # 5 strategies × 2 modes = 10 configurations
    assert len(runs) == 10
    assert mock_build.call_count == 10
    assert mock_run_eval.call_count == 10
    # Corrective flag is threaded through
    corrective_flags = [c.kwargs.get("use_corrective") for c in mock_run_eval.call_args_list]
    assert corrective_flags.count(True) == 5
    assert corrective_flags.count(False) == 5


def test_run_ablation_respects_explicit_strategies_and_modes() -> None:
    dummy_summary = _make_summary([_make_result("c1", relevancy=0.9, correctness=0.9)])

    with (
        patch("rag_harness.evaluation.ablation.build_retriever", return_value=MagicMock()),
        patch(
            "rag_harness.evaluation.ablation.run_eval",
            new_callable=AsyncMock,
            return_value=dummy_summary,
        ) as mock_run_eval,
    ):
        runs = asyncio.run(run_ablation(strategies=["dense", "hybrid"], corrective_modes=[False]))

    assert len(runs) == 2
    assert {r.strategy for r in runs} == {"dense", "hybrid"}
    assert all(not r.corrective for r in runs)
    assert mock_run_eval.call_count == 2


def test_run_ablation_skips_unknown_strategy() -> None:
    dummy_summary = _make_summary([_make_result("c1", relevancy=0.9, correctness=0.9)])

    with (
        patch("rag_harness.evaluation.ablation.build_retriever", return_value=MagicMock()),
        patch(
            "rag_harness.evaluation.ablation.run_eval",
            new_callable=AsyncMock,
            return_value=dummy_summary,
        ),
    ):
        runs = asyncio.run(run_ablation(strategies=["dense", "bogus"], corrective_modes=[False]))

    assert len(runs) == 1
    assert runs[0].strategy == "dense"


def test_run_ablation_records_rbi_count_and_rate() -> None:
    results = [
        _make_result("hit", relevancy=0.9, correctness=0.2),  # RBI
        _make_result("clean", relevancy=0.9, correctness=0.9),
    ]
    summary = _make_summary(results)

    with (
        patch("rag_harness.evaluation.ablation.build_retriever", return_value=MagicMock()),
        patch(
            "rag_harness.evaluation.ablation.run_eval", new_callable=AsyncMock, return_value=summary
        ),
    ):
        runs = asyncio.run(run_ablation(strategies=["dense"], corrective_modes=[False]))

    assert runs[0].rbi_count == 1
    assert runs[0].rbi_rate == 0.5


def test_run_ablation_continues_after_a_configuration_raises() -> None:
    dummy_summary = _make_summary([_make_result("c1", relevancy=0.9, correctness=0.9)])

    call_count = {"n": 0}

    def _run_eval_side(*args, **kwargs):
        call_count["n"] += 1
        # Second call raises; others succeed
        if call_count["n"] == 2:
            raise RuntimeError("second config broke")
        return dummy_summary

    with (
        patch("rag_harness.evaluation.ablation.build_retriever", return_value=MagicMock()),
        patch(
            "rag_harness.evaluation.ablation.run_eval",
            new_callable=AsyncMock,
            side_effect=_run_eval_side,
        ),
    ):
        runs = asyncio.run(
            run_ablation(strategies=["dense", "hybrid", "hyde"], corrective_modes=[False])
        )

    # Second config skipped, others recorded
    assert len(runs) == 2
    assert {r.strategy for r in runs} == {"dense", "hyde"}


# --- render_markdown ---


def test_render_markdown_returns_table_with_expected_headers() -> None:
    dummy_summary = _make_summary([_make_result("c1", relevancy=0.9, correctness=0.9)])
    run = AblationRun(
        strategy="dense",
        corrective=False,
        summary=dummy_summary,
        timestamp="2026-07-03T00:00:00+00:00",
        git_commit="abc1234",
        rbi_count=0,
        rbi_rate=0.0,
    )
    md = render_markdown([run])
    for header in [
        "Strategy",
        "Corrective",
        "Recall",
        "Precision",
        "Faith",
        "Correct",
        "Relevancy",
        "Rel-but-Incorrect",
        "Cost",
        "p50 ms",
        "p95 ms",
    ]:
        assert header in md
    assert "dense" in md
    # Body line count = 1 config
    assert md.count("| dense |") == 1


def test_render_markdown_empty_runs() -> None:
    md = render_markdown([])
    assert "No successful" in md


# --- render_csv ---


def test_render_csv_emits_one_row_per_case_per_configuration() -> None:
    r1 = _make_result("c1", relevancy=0.9, correctness=0.2)
    r2 = _make_result("c2", relevancy=0.9, correctness=0.9)
    summary = _make_summary([r1, r2])
    run_dense = AblationRun(
        strategy="dense",
        corrective=False,
        summary=summary,
        timestamp="ts",
        git_commit="sha",
        rbi_count=1,
        rbi_rate=0.5,
    )
    run_hybrid = AblationRun(
        strategy="hybrid",
        corrective=True,
        summary=summary,
        timestamp="ts",
        git_commit="sha",
        rbi_count=1,
        rbi_rate=0.5,
    )
    csv_text = render_csv([run_dense, run_hybrid])

    # 2 configs × 2 cases = 4 data rows + 1 header
    assert csv_text.count("\n") == 5
    assert "is_relevant_but_incorrect" in csv_text
    # RBI flag correctly emitted per row
    assert "True" in csv_text  # c1 is RBI
    assert "False" in csv_text  # c2 is not


def test_render_csv_roundtrips_through_dictreader(tmp_path: Path) -> None:
    import csv as _csv

    r = _make_result("c1", relevancy=0.9, correctness=0.2)
    summary = _make_summary([r])
    run = AblationRun(
        strategy="dense",
        corrective=False,
        summary=summary,
        timestamp="ts",
        git_commit="sha",
        rbi_count=1,
        rbi_rate=1.0,
    )
    csv_text = render_csv([run])
    reader = _csv.DictReader(csv_text.splitlines())
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c1"
    assert rows[0]["strategy"] == "dense"
    assert rows[0]["is_relevant_but_incorrect"] == "True"
