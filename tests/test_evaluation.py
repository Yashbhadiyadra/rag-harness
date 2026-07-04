import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_harness.evaluation.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
)
from rag_harness.evaluation.runner import (
    _percentile,
    export_results,
    load_golden_cases,
    run_eval,
)
from rag_harness.models import Chunk, EvalResult, EvalSummary


def _make_chunk(source_file: str) -> Chunk:
    return Chunk(
        id=f"{source_file}::0",
        text="Some content.",
        source_file=source_file,
        git_commit="abc123",
        doc_version="v1.29",
        chunk_index=0,
        heading_path=[],
    )


# --- context_recall (deterministic, no mocks needed) ---


def test_context_recall_full_hit() -> None:
    chunks = [_make_chunk("content/en/docs/security/rbac.md")]
    score = context_recall(chunks, ["content/en/docs/security/rbac.md"])
    assert score == 1.0


def test_context_recall_partial_hit() -> None:
    chunks = [_make_chunk("content/en/docs/security/rbac.md")]
    score = context_recall(
        chunks,
        [
            "content/en/docs/security/rbac.md",
            "content/en/docs/concepts/overview.md",
        ],
    )
    assert score == 0.5


def test_context_recall_no_hit() -> None:
    chunks = [_make_chunk("content/en/docs/networking/services.md")]
    score = context_recall(chunks, ["content/en/docs/security/rbac.md"])
    assert score == 0.0


def test_context_recall_empty_relevant() -> None:
    # No relevant docs specified → perfect score by convention
    score = context_recall([], [])
    assert score == 1.0


# --- load_golden_cases ---


def test_load_golden_cases(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "test.json").write_text(json.dumps(data))
    cases = load_golden_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].id == "test-001"


# --- run_eval gate ---


def test_run_eval_passes_when_above_thresholds(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_make_chunk("content/en/docs/security/rbac.md")]

    with (
        patch("rag_harness.evaluation.runner.generate", return_value="Role-Based Access Control."),
        patch("rag_harness.evaluation.runner.faithfulness", return_value=0.95),
        patch("rag_harness.evaluation.runner.correctness", return_value=0.90),
        patch("rag_harness.evaluation.runner.answer_relevancy", return_value=0.9),
        patch("rag_harness.evaluation.runner.context_precision", return_value=0.8),
    ):
        summary = run_eval(mock_retriever, golden_dir=tmp_path)

    assert summary.passed is True
    assert summary.mean_context_recall == 1.0
    # New metrics carried through into summary
    assert summary.mean_answer_relevancy == 0.9
    assert summary.mean_context_precision == 0.8


def test_run_eval_fails_when_below_threshold(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []  # retrieval miss → recall=0.0

    with (
        patch("rag_harness.evaluation.runner.generate", return_value="I do not know."),
        patch("rag_harness.evaluation.runner.faithfulness", return_value=0.90),
        patch("rag_harness.evaluation.runner.correctness", return_value=0.80),
        patch("rag_harness.evaluation.runner.answer_relevancy", return_value=0.5),
        patch("rag_harness.evaluation.runner.context_precision", return_value=0.0),
    ):
        summary = run_eval(mock_retriever, golden_dir=tmp_path)

    assert summary.passed is False
    assert summary.mean_context_recall == 0.0


# --- export_results ---


def _make_summary() -> EvalSummary:
    result = EvalResult(
        case_id="test-001",
        question="What is RBAC?",
        generated_answer="Role-Based Access Control.",
        retrieved_doc_ids=["content/en/docs/security/rbac.md"],
        context_recall=1.0,
        faithfulness=0.95,
        correctness=0.90,
    )
    return EvalSummary(
        results=[result],
        mean_context_recall=1.0,
        mean_faithfulness=0.95,
        mean_correctness=0.90,
        passed=True,
    )


def test_export_results_json(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    export_results(_make_summary(), out)
    data = json.loads(out.read_text())
    assert data["passed"] is True
    assert data["mean_context_recall"] == 1.0
    # New operational fields are included in the JSON export
    assert "latency_p50_ms" in data
    assert "latency_p95_ms" in data
    assert "total_cost_usd" in data
    assert "mean_answer_relevancy" in data
    assert "mean_context_precision" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["case_id"] == "test-001"


def test_export_results_csv(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    export_results(_make_summary(), out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "test-001"
    assert float(rows[0]["context_recall"]) == 1.0
    assert float(rows[0]["faithfulness"]) == 0.95


def test_export_results_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        export_results(_make_summary(), tmp_path / "results.txt")


# --- answer_relevancy ---


def _mock_llm_score(score: str) -> MagicMock:
    """Return a mock OpenAI client whose chat completion returns a raw score string."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = score
    resp.usage = MagicMock(prompt_tokens=50, completion_tokens=3)
    client.chat.completions.create.return_value = resp
    return client


def test_answer_relevancy_parses_score() -> None:
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("0.85")):
        score = answer_relevancy("What is RBAC?", "RBAC controls permissions.")
    assert score == 0.85


def test_answer_relevancy_empty_answer_returns_zero() -> None:
    # No LLM call needed for empty answer — short-circuit
    with patch("rag_harness.evaluation.metrics._client") as mock_client:
        score = answer_relevancy("What is RBAC?", "")
    assert score == 0.0
    mock_client.chat.completions.create.assert_not_called()


def test_answer_relevancy_whitespace_only_returns_zero() -> None:
    with patch("rag_harness.evaluation.metrics._client") as mock_client:
        score = answer_relevancy("What is RBAC?", "   \n   ")
    assert score == 0.0
    mock_client.chat.completions.create.assert_not_called()


def test_answer_relevancy_clamps_out_of_range() -> None:
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("1.7")):
        assert answer_relevancy("Q?", "A.") == 1.0
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("-0.3")):
        assert answer_relevancy("Q?", "A.") == 0.0


# --- context_precision ---


def test_context_precision_parses_score() -> None:
    chunks = [_make_chunk("content/en/docs/security/rbac.md")]
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("0.6")):
        score = context_precision("What is RBAC?", chunks, "RBAC controls permissions.")
    assert score == 0.6


def test_context_precision_empty_chunks_returns_zero() -> None:
    with patch("rag_harness.evaluation.metrics._client") as mock_client:
        score = context_precision("Q?", [], "reference")
    assert score == 0.0
    # No LLM call when there are no chunks to score
    mock_client.chat.completions.create.assert_not_called()


def test_context_precision_includes_all_chunks_in_prompt() -> None:
    chunks = [
        _make_chunk("docs/a.md"),
        _make_chunk("docs/b.md"),
        _make_chunk("docs/c.md"),
    ]
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("0.5")) as mc:
        context_precision("Q?", chunks, "ref")
    # Verify all chunk texts appear in the LLM prompt (order doesn't matter for this test)
    call_kwargs = mc.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][1]["content"]
    assert "[1]" in user_msg and "[2]" in user_msg and "[3]" in user_msg


def test_context_precision_handles_non_numeric_response() -> None:
    chunks = [_make_chunk("docs/a.md")]
    with patch(
        "rag_harness.evaluation.metrics._client",
        _mock_llm_score("this is not a number"),
    ):
        score = context_precision("Q?", chunks, "ref")
    # Falls back to 0.0 per the LLM judge parsing contract
    assert score == 0.0


# --- _percentile ---


def test_percentile_empty_list_returns_zero() -> None:
    assert _percentile([], 50) == 0.0


def test_percentile_single_element() -> None:
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 95) == 42.0


def test_percentile_odd_length_p50() -> None:
    # 5 elements: nearest-rank rank(50) = ceil(0.5 * 5) = 3 → index 2 → 30
    assert _percentile([10, 20, 30, 40, 50], 50) == 30


def test_percentile_p95_on_20_samples() -> None:
    # 20 elements 1..20: rank(95) = ceil(0.95 * 20) = 19 → index 18 → 19
    assert _percentile([float(i) for i in range(1, 21)], 95) == 19


def test_percentile_unsorted_input() -> None:
    # Percentile is computed on sorted values regardless of input order
    assert _percentile([50, 10, 30, 40, 20], 50) == 30


# --- run_eval operational aggregation ---


def test_run_eval_aggregates_operational_metrics(tmp_path: Path) -> None:
    data = [
        {
            "id": f"test-{i}",
            "question": f"Q{i}?",
            "reference_answer": f"A{i}.",
            "relevant_doc_ids": ["docs/a.md"],
        }
        for i in range(3)
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_make_chunk("docs/a.md")]

    # Simulate a real call recording usage — the evaluate_case wrapper opens
    # its own collect_usage() block, so we need something inside to record.
    # Easiest: patch the metric functions to return numbers but also stub the
    # generation LLM path to inject a usage record via the ContextVar.

    from rag_harness.observability.usage import TokenUsage, record_usage

    def _generate_stub(question: str, chunks: list) -> str:
        # Each evaluate_case call records one 100-token record via generate
        record_usage(TokenUsage("gpt-4o-mini", 100, 20, 0.030))
        return f"answer for {question}"

    with (
        patch("rag_harness.evaluation.runner.generate", side_effect=_generate_stub),
        patch("rag_harness.evaluation.runner.faithfulness", return_value=0.9),
        patch("rag_harness.evaluation.runner.correctness", return_value=0.9),
        patch("rag_harness.evaluation.runner.answer_relevancy", return_value=0.9),
        patch("rag_harness.evaluation.runner.context_precision", return_value=0.9),
    ):
        summary = run_eval(mock_retriever, golden_dir=tmp_path)

    # 3 cases × (100 in, 20 out, $0.030)
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 60
    assert round(summary.total_cost_usd, 6) == 0.090
    # Latency was measured (non-zero) but exact value depends on wall clock
    assert summary.latency_p50_ms > 0.0
    assert summary.latency_p95_ms >= summary.latency_p50_ms
