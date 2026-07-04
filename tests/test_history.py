"""Unit tests for the append-only eval history JSONL."""

import json
from pathlib import Path
from unittest.mock import patch

from rag_harness.evaluation.history import (
    HistoryEntry,
    _current_git_commit,
    load_history,
    record_run,
)
from rag_harness.models import EvalResult, EvalSummary


def _make_summary(mean_recall: float = 0.85) -> EvalSummary:
    result = EvalResult(
        case_id="test-001",
        question="Q?",
        generated_answer="A.",
        retrieved_doc_ids=["docs/a.md"],
        context_recall=mean_recall,
        context_precision=0.7,
        faithfulness=0.9,
        correctness=0.8,
        answer_relevancy=0.85,
        latency_ms=200.0,
        input_tokens=100,
        output_tokens=50,
        estimated_cost_usd=0.001,
    )
    return EvalSummary(
        results=[result],
        mean_context_recall=mean_recall,
        mean_context_precision=0.7,
        mean_faithfulness=0.9,
        mean_correctness=0.8,
        mean_answer_relevancy=0.85,
        latency_p50_ms=200.0,
        latency_p95_ms=250.0,
        total_cost_usd=0.001,
        total_input_tokens=100,
        total_output_tokens=50,
        passed=True,
    )


def test_record_run_writes_one_json_line(tmp_path: Path) -> None:
    history_file = tmp_path / "runs.jsonl"
    entry = record_run(_make_summary(), "dense", False, history_file=history_file)

    lines = history_file.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["strategy"] == "dense"
    assert parsed["corrective"] is False
    assert parsed["mean_context_recall"] == 0.85
    assert parsed["passed"] is True
    assert isinstance(entry, HistoryEntry)


def test_record_run_appends_without_modifying_existing_lines(tmp_path: Path) -> None:
    history_file = tmp_path / "runs.jsonl"
    record_run(_make_summary(mean_recall=0.5), "dense", False, history_file=history_file)
    record_run(_make_summary(mean_recall=0.9), "hybrid", True, history_file=history_file)

    lines = history_file.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["strategy"] == "dense"
    assert json.loads(lines[0])["mean_context_recall"] == 0.5
    assert json.loads(lines[1])["strategy"] == "hybrid"
    assert json.loads(lines[1])["mean_context_recall"] == 0.9


def test_record_run_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "path" / "runs.jsonl"
    record_run(_make_summary(), "dense", False, history_file=nested)
    assert nested.exists()


def test_load_history_returns_entries_in_order(tmp_path: Path) -> None:
    history_file = tmp_path / "runs.jsonl"
    record_run(_make_summary(mean_recall=0.1), "s1", False, history_file=history_file)
    record_run(_make_summary(mean_recall=0.2), "s2", True, history_file=history_file)
    record_run(_make_summary(mean_recall=0.3), "s3", False, history_file=history_file)

    entries = load_history(history_file=history_file)
    assert [e.strategy for e in entries] == ["s1", "s2", "s3"]
    assert [e.mean_context_recall for e in entries] == [0.1, 0.2, 0.3]


def test_load_history_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_history(history_file=tmp_path / "does_not_exist.jsonl") == []


def test_load_history_skips_malformed_lines(tmp_path: Path) -> None:
    history_file = tmp_path / "runs.jsonl"
    # Two valid lines with a corrupted one in the middle
    record_run(_make_summary(mean_recall=0.5), "s1", False, history_file=history_file)
    with history_file.open("a") as fh:
        fh.write("this is not json\n")
    record_run(_make_summary(mean_recall=0.7), "s2", False, history_file=history_file)

    entries = load_history(history_file=history_file)
    assert [e.strategy for e in entries] == ["s1", "s2"]


def test_load_history_skips_comment_lines(tmp_path: Path) -> None:
    history_file = tmp_path / "runs.jsonl"
    with history_file.open("w") as fh:
        fh.write("# This is a header comment\n")
        fh.write("\n")  # blank line
    record_run(_make_summary(), "dense", False, history_file=history_file)

    entries = load_history(history_file=history_file)
    assert len(entries) == 1
    assert entries[0].strategy == "dense"


def test_current_git_commit_handles_missing_git() -> None:
    # Simulate git being unavailable via subprocess.check_output raising
    with patch(
        "rag_harness.evaluation.history.subprocess.check_output",
        side_effect=FileNotFoundError(),
    ):
        assert _current_git_commit() == "unknown"


def test_current_git_commit_handles_non_git_directory() -> None:
    from subprocess import CalledProcessError

    with patch(
        "rag_harness.evaluation.history.subprocess.check_output",
        side_effect=CalledProcessError(returncode=128, cmd=["git"]),
    ):
        assert _current_git_commit() == "unknown"
