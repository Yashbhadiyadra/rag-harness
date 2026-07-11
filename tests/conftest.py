"""Shared pytest fixtures and environment setup."""

import os
from pathlib import Path

import pytest

# Set before any rag_harness module is imported - config.py instantiates Settings()
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


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Clear the in-memory rate-limiter state between tests.

    Without this, the tightened public-demo limits (10/hour;3/minute per IP,
    see ADR-0010) carry state across tests and cause spurious 429s in any
    test file that exercises /query via TestClient.
    """
    from rag_harness.api.server import app

    app.state.limiter.reset()


@pytest.fixture(autouse=True)
def _reset_daily_budget() -> None:
    """Clear the global daily-budget counter between tests.

    The demo daily cap (ADR-0010) is enforced by an in-memory counter on
    ``app.state.daily_budget``. Every /query hit consumes one slot, so
    without this fixture tests accumulate consumption and later tests can
    fail with spurious 429 / demo_daily_limit_reached responses.
    """
    from rag_harness.api.server import app

    app.state.daily_budget.reset()
