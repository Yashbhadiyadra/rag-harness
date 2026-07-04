"""Append-only JSONL history of every eval and ablation run.

Each line in ``evals/history/runs.jsonl`` is one ``HistoryEntry``: timestamp,
git commit SHA, strategy, corrective flag, all summary metrics. The file is
committed to the repo so trends over time are visible on the commit graph
and regressions are attributable to specific commits.

The file grows one line per ``run_eval`` invocation and one line per
configuration in an ablation. A 30-case, 10-config ablation adds 10 lines.
Prunable later if it becomes an issue.
"""

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from rag_harness.models import EvalSummary

logger = logging.getLogger(__name__)

_HISTORY_DIR = Path(__file__).parent.parent.parent.parent / "evals" / "history"
_HISTORY_FILE = _HISTORY_DIR / "runs.jsonl"


class HistoryEntry(BaseModel):
    """One row in the append-only eval history."""

    timestamp: str  # ISO-8601 UTC
    git_commit: str
    strategy: str
    corrective: bool
    n_cases: int
    passed: bool
    mean_context_recall: float
    mean_context_precision: float
    mean_faithfulness: float
    mean_correctness: float
    mean_answer_relevancy: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_cost_usd: float


def _current_git_commit() -> str:
    """Return the short SHA of HEAD; ``'unknown'`` if git is unavailable.

    Duplicated in ablation.py; kept local to avoid a circular import between
    history and ablation.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def record_run(
    summary: EvalSummary,
    strategy: str,
    corrective: bool,
    history_file: Path | None = None,
) -> HistoryEntry:
    """Append one line to the eval history JSONL and return the entry written.

    Idempotent by convention only — every call appends a new line. Callers
    should invoke exactly once per ``run_eval`` invocation or per ablation
    configuration.
    """
    path = history_file or _HISTORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = HistoryEntry(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_current_git_commit(),
        strategy=strategy,
        corrective=corrective,
        n_cases=len(summary.results),
        passed=summary.passed,
        mean_context_recall=summary.mean_context_recall,
        mean_context_precision=summary.mean_context_precision,
        mean_faithfulness=summary.mean_faithfulness,
        mean_correctness=summary.mean_correctness,
        mean_answer_relevancy=summary.mean_answer_relevancy,
        latency_p50_ms=summary.latency_p50_ms,
        latency_p95_ms=summary.latency_p95_ms,
        total_cost_usd=summary.total_cost_usd,
    )

    with path.open("a") as fh:
        fh.write(entry.model_dump_json() + "\n")

    logger.debug("appended history entry: %s corrective=%s", strategy, corrective)
    return entry


def load_history(history_file: Path | None = None) -> list[HistoryEntry]:
    """Read every line of the history file and return parsed entries in order.

    Missing file returns an empty list. Malformed lines are skipped with a
    warning rather than aborting — a corrupted line at the tail should not
    hide the intact history above it.
    """
    path = history_file or _HISTORY_FILE
    if not path.exists():
        return []

    entries: list[HistoryEntry] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entries.append(HistoryEntry.model_validate_json(line))
        except Exception as e:
            logger.warning("skipping malformed history line %d: %s", i, e)
    return entries
