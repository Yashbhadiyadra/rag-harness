import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_harness.evaluation.metrics import context_recall
from rag_harness.evaluation.runner import export_results, load_golden_cases, run_eval
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
    ):
        summary = run_eval(mock_retriever, golden_dir=tmp_path)

    assert summary.passed is True
    assert summary.mean_context_recall == 1.0


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
