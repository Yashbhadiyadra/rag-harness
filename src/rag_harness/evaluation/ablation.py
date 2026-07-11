"""Ablation runner - golden set × strategy × corrective-mode → comparative table.

Runs the golden eval suite across every retrieval strategy in
``VALID_STRATEGIES`` × {baseline, corrective} = 10 configurations by default.
Emits three artifacts per run:

  1. Markdown table (for a README, an interviewer, or the metrics page).
  2. CSV export with a full per-case ``is_relevant_but_incorrect`` column
     (for a spreadsheet review).
  3. One JSON summary object per configuration (persisted downstream by the
     history layer).

The **relevant-but-incorrect** category is a first-class output, not a
footnote. A case scoring high on ``answer_relevancy`` but low on
``correctness`` is a confident-sounding hallucination - the answer addresses
the question but gets the facts wrong. That failure mode is more dangerous
than an off-topic answer and gets its own column in the markdown, its own
line in the summary, and a per-case flag in the CSV.
"""

import csv
import io
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from rag_harness.config import settings
from rag_harness.evaluation.history import record_run
from rag_harness.evaluation.runner import run_eval
from rag_harness.models import EvalResult, EvalSummary
from rag_harness.retrieval.factory import VALID_STRATEGIES, build_retriever

logger = logging.getLogger(__name__)

# Preferred display order for the ablation table (dense → most sophisticated)
_STRATEGY_ORDER = ["dense", "hybrid", "hybrid-rerank", "hyde", "full"]


class AblationRun(BaseModel):
    """One configuration's outcome: strategy × corrective × summary + metadata."""

    strategy: str
    corrective: bool
    summary: EvalSummary
    timestamp: str  # ISO-8601 UTC
    git_commit: str
    rbi_count: int  # relevant-but-incorrect count
    rbi_rate: float  # relevant-but-incorrect rate in [0, 1]


def _current_git_commit() -> str:
    """Return the short SHA of HEAD; ``'unknown'`` if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def is_relevant_but_incorrect(
    result: EvalResult,
    relevancy_min: float | None = None,
    correctness_max: float | None = None,
) -> bool:
    """Return True if *result* is the highlighted 'relevant but incorrect' failure.

    Definition: ``answer_relevancy > relevancy_min`` AND
    ``correctness < correctness_max`` - a confident, on-topic answer that gets
    the facts wrong. Thresholds default to settings values.
    """
    rmin = relevancy_min if relevancy_min is not None else settings.rbi_relevancy_min
    cmax = correctness_max if correctness_max is not None else settings.rbi_correctness_max
    return result.answer_relevancy > rmin and result.correctness < cmax


def relevant_but_incorrect_cases(summary: EvalSummary) -> list[EvalResult]:
    """Return every case in *summary* that matches the RBI predicate."""
    return [r for r in summary.results if is_relevant_but_incorrect(r)]


async def run_ablation(
    strategies: list[str] | None = None,
    corrective_modes: list[bool] | None = None,
    golden_dir: Path | None = None,
) -> list[AblationRun]:
    """Run the eval suite across every requested configuration.

    Defaults: every strategy in ``VALID_STRATEGIES`` × [False, True]. One
    configuration that raises does not abort the whole run - the exception
    is logged and the remaining configurations proceed. Configurations that
    succeed are returned in strategy-then-corrective order.
    """
    strats = strategies if strategies is not None else _STRATEGY_ORDER
    modes = corrective_modes if corrective_modes is not None else [False, True]
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    git_sha = _current_git_commit()

    runs: list[AblationRun] = []
    for strategy in strats:
        if strategy not in VALID_STRATEGIES:
            logger.warning("skipping unknown strategy %r", strategy)
            continue
        for corrective in modes:
            logger.info("ablation: strategy=%s corrective=%s", strategy, corrective)
            try:
                retriever = build_retriever(strategy)
                summary = await run_eval(
                    retriever,
                    golden_dir=golden_dir,
                    use_corrective=corrective,
                    strategy_label=strategy,
                    # Ablation writes its own history entry per config, so
                    # skip the run_eval-side write to avoid double records.
                    record_history=False,
                )
                record_run(summary, strategy=strategy, corrective=corrective)
            except Exception as e:
                logger.error(
                    "ablation configuration failed (strategy=%s corrective=%s): %s",
                    strategy,
                    corrective,
                    e,
                )
                continue
            rbi = relevant_but_incorrect_cases(summary)
            runs.append(
                AblationRun(
                    strategy=strategy,
                    corrective=corrective,
                    summary=summary,
                    timestamp=timestamp,
                    git_commit=git_sha,
                    rbi_count=len(rbi),
                    rbi_rate=len(rbi) / len(summary.results) if summary.results else 0.0,
                )
            )
    return runs


def render_markdown(runs: list[AblationRun]) -> str:
    """Render the ablation runs as a compact markdown table with an RBI column."""
    if not runs:
        return "_No successful ablation configurations._\n"

    header = (
        "| Strategy | Corrective | Recall | Precision | Faith | Correct | Relevancy | "
        "Rel-but-Incorrect | Cost | p50 ms | p95 ms |\n"
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for r in runs:
        s = r.summary
        rows.append(
            f"| {r.strategy} | {'yes' if r.corrective else 'no'} | "
            f"{s.mean_context_recall:.2f} | {s.mean_context_precision:.2f} | "
            f"{s.mean_faithfulness:.2f} | {s.mean_correctness:.2f} | "
            f"{s.mean_answer_relevancy:.2f} | "
            f"{r.rbi_count} ({r.rbi_rate:.0%}) | "
            f"${s.total_cost_usd:.4f} | "
            f"{s.latency_p50_ms:.0f} | {s.latency_p95_ms:.0f} |"
        )
    return header + "\n".join(rows) + "\n"


def render_csv(runs: list[AblationRun]) -> str:
    """Render every case across every configuration as a single CSV.

    One row per (configuration, case). Adds an ``is_relevant_but_incorrect``
    column so a reviewer can filter to just the highlighted failures.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "strategy",
            "corrective",
            "case_id",
            "question",
            "context_recall",
            "context_precision",
            "faithfulness",
            "correctness",
            "answer_relevancy",
            "is_relevant_but_incorrect",
            "latency_ms",
            "estimated_cost_usd",
            "corrective_category",
            "corrective_attempts",
        ],
    )
    writer.writeheader()
    for r in runs:
        for case in r.summary.results:
            writer.writerow(
                {
                    "strategy": r.strategy,
                    "corrective": r.corrective,
                    "case_id": case.case_id,
                    "question": case.question,
                    "context_recall": case.context_recall,
                    "context_precision": case.context_precision,
                    "faithfulness": case.faithfulness,
                    "correctness": case.correctness,
                    "answer_relevancy": case.answer_relevancy,
                    "is_relevant_but_incorrect": is_relevant_but_incorrect(case),
                    "latency_ms": case.latency_ms,
                    "estimated_cost_usd": case.estimated_cost_usd,
                    "corrective_category": case.corrective_category,
                    "corrective_attempts": case.corrective_attempts,
                }
            )
    return buf.getvalue()
