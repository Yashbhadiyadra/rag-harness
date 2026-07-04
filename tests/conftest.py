"""Shared pytest fixtures and environment setup."""

import os
from pathlib import Path

import pytest

# Set before any rag_harness module is imported — config.py instantiates Settings()
# at module level, so the env var must exist before collection begins.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")


@pytest.fixture(autouse=True)
def _isolate_eval_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the eval history file to tmp_path for every test.

    Without this, any test that calls run_eval or run_ablation (even fully
    mocked) would append lines to the real evals/history/runs.jsonl.
    """
    fake_history = tmp_path / "test_runs.jsonl"
    monkeypatch.setattr("rag_harness.evaluation.history._HISTORY_FILE", fake_history)
    yield fake_history
