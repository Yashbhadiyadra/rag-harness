"""Judge reliability audit - measure the judges before trusting their scores.

Implements ADR-0014. The core experiment: score a golden reference answer
against itself (a calibrated correctness judge should return ~1.0), then
apply meaning-preserving format perturbations and re-score. The semantic
content is identical by construction, so any score shift is pure format
sensitivity - the failure mode arXiv:2603.05399 found degrades judges
more than semantic changes do.

Pure statistics (perturbations, shift stats, flip rate, Cohen's kappa)
are separated from the LLM-calling runner so they test without a network.
"""

import asyncio
import hashlib
import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rag_harness.config import settings
from rag_harness.evaluation.metrics import (
    _ANSWER_RELEVANCY_PROMPT,
    _CORRECTNESS_PROMPT,
    answer_relevancy_async,
    correctness_async,
)
from rag_harness.evaluation.runner import load_golden_cases
from rag_harness.models import GoldenCase

logger = logging.getLogger(__name__)

# --- Format perturbations (deterministic, meaning-preserving) ----------


def _code_fence(text: str) -> str:
    """Wrap the whole answer in a fenced code block."""
    return f"```text\n{text}\n```"


def _bullets(text: str) -> str:
    """Recast sentences as a bullet list. Meaning is untouched."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return "\n".join(f"- {s if s.endswith('.') else s + '.'}" for s in sentences)


def _whitespace(text: str) -> str:
    """Inflate whitespace: double internal spaces, pad with blank lines."""
    return "\n\n" + text.replace(" ", "  ") + "\n\n\n"


def _bold_lead(text: str) -> str:
    """Bold the first sentence, as chat models often do."""
    head, sep, tail = text.partition(". ")
    if not sep:
        return f"**{text}**"
    return f"**{head}.** {tail}"


PERTURBATIONS: dict[str, Callable[[str], str]] = {
    "code_fence": _code_fence,
    "bullets": _bullets,
    "whitespace": _whitespace,
    "bold_lead": _bold_lead,
}


# --- Pure statistics ----------------------------------------------------


@dataclass
class ShiftStats:
    """Score-shift summary for one (metric, perturbation) pair."""

    perturbation: str
    n: int
    mean_abs_shift: float
    max_abs_shift: float
    flip_rate: float  # fraction of cases whose gate verdict flipped


def score_shift_stats(
    base: list[float],
    perturbed: list[float],
    threshold: float,
    perturbation: str,
) -> ShiftStats:
    """Summarise how far perturbed scores moved from their base scores.

    ``threshold`` is the reliability-gate cut for this metric; a "flip" is
    a case whose pass/fail verdict changes purely because of formatting.
    """
    if len(base) != len(perturbed) or not base:
        raise ValueError("base and perturbed must be equal-length, non-empty lists")
    deltas = [abs(b - p) for b, p in zip(base, perturbed, strict=True)]
    flips = sum(
        1 for b, p in zip(base, perturbed, strict=True) if (b >= threshold) != (p >= threshold)
    )
    return ShiftStats(
        perturbation=perturbation,
        n=len(base),
        mean_abs_shift=sum(deltas) / len(deltas),
        max_abs_shift=max(deltas),
        flip_rate=flips / len(base),
    )


def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Chance-corrected agreement between two binary label lists.

    Reported instead of raw percent agreement, which overstates judge
    quality by 34-41pp on public benchmarks (arXiv:2606.19544). Returns
    1.0 for perfect agreement; 0.0 when agreement equals chance. When
    both raters are constant (chance agreement is 1.0), kappa is
    undefined; returns 1.0 if they agree everywhere, else 0.0.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("label lists must be equal-length and non-empty")
    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b, strict=True) if a == b) / n
    p_a = sum(labels_a) / n
    p_b = sum(labels_b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


# --- Provenance ---------------------------------------------------------


def prompt_hash(prompt: str) -> str:
    """Short SHA-256 of a judge prompt - makes silent prompt drift detectable."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _current_git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# --- Audit runner --------------------------------------------------------


@dataclass
class AuditReport:
    """Full result of one judge-reliability audit run."""

    judge_model: str
    commit: str
    timestamp: str
    n_cases: int
    prompt_hashes: dict[str, str]
    base_mean: dict[str, float]  # metric -> mean base (unperturbed) score
    shifts: dict[str, list[ShiftStats]] = field(default_factory=dict)  # metric -> stats


async def _score_case(metric: str, case: GoldenCase, answer: str) -> float:
    if metric == "correctness":
        return await correctness_async(case.question, answer, case.reference_answer)
    if metric == "answer_relevancy":
        return await answer_relevancy_async(case.question, answer)
    raise ValueError(f"unknown audit metric: {metric}")


_METRIC_THRESHOLDS: dict[str, float] = {
    # correctness shares the reliability-gate cut; relevancy has no gate,
    # so use the ablation's relevant-but-incorrect boundary as the verdict line.
    "correctness": settings.threshold_correctness,
    "answer_relevancy": settings.rbi_relevancy_min,
}


async def run_audit(
    cases: list[GoldenCase] | None = None,
    metrics: tuple[str, ...] = ("correctness", "answer_relevancy"),
) -> AuditReport:
    """Run the format-invariance audit over the golden set.

    For each case the *reference answer itself* is judged (base), then each
    perturbed variant. Base correctness should be ~1.0 - the reference
    compared against itself - so both the calibration gap (1.0 - base mean)
    and the perturbation shifts are diagnostic.
    """
    if cases is None:
        cases = load_golden_cases()
    report = AuditReport(
        judge_model=settings.generation_model,
        commit=_current_git_commit(),
        timestamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%S+0000"),
        n_cases=len(cases),
        prompt_hashes={
            "correctness": prompt_hash(_CORRECTNESS_PROMPT),
            "answer_relevancy": prompt_hash(_ANSWER_RELEVANCY_PROMPT),
        },
        base_mean={},
    )

    for metric in metrics:
        threshold = _METRIC_THRESHOLDS[metric]
        base_scores = [await _score_case(metric, c, c.reference_answer) for c in cases]
        report.base_mean[metric] = sum(base_scores) / len(base_scores)
        stats: list[ShiftStats] = []
        for name, perturb in PERTURBATIONS.items():
            perturbed_scores = [
                await _score_case(metric, c, perturb(c.reference_answer)) for c in cases
            ]
            stats.append(score_shift_stats(base_scores, perturbed_scores, threshold, name))
            logger.info(
                "audit %s/%s: mean shift %.3f, flips %.0f%%",
                metric,
                name,
                stats[-1].mean_abs_shift,
                stats[-1].flip_rate * 100,
            )
        report.shifts[metric] = stats
    return report


# --- Report rendering ----------------------------------------------------


def render_markdown(report: AuditReport) -> str:
    """Render the audit report in the experiments-file house style."""
    lines = [
        "# Judge reliability audit (ADR-0014)",
        "",
        f"- Judge model: `{report.judge_model}`",
        f"- Commit: `{report.commit}`  ·  Timestamp: {report.timestamp}",
        f"- Golden cases: {report.n_cases}",
        "- Prompt hashes: " + ", ".join(f"{m} `{h}`" for m, h in report.prompt_hashes.items()),
        "",
        "Base = reference answer judged as-is (a calibrated correctness judge",
        "returns ~1.0). Perturbations are meaning-preserving; any shift is",
        "pure format sensitivity. Flip rate = verdicts changed at the gate",
        "threshold by formatting alone.",
        "",
    ]
    for metric, stats in report.shifts.items():
        lines.append(f"## {metric}")
        lines.append("")
        lines.append(f"Base mean score: **{report.base_mean[metric]:.3f}**")
        lines.append("")
        lines.append("| Perturbation | n | Mean abs shift | Max abs shift | Flip rate |")
        lines.append("|---|---|---|---|---|")
        for s in stats:
            lines.append(
                f"| {s.perturbation} | {s.n} | {s.mean_abs_shift:.3f} "
                f"| {s.max_abs_shift:.3f} | {s.flip_rate:.0%} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_report(report: AuditReport, out_dir: Path | None = None) -> Path:
    """Write markdown + JSON to evals/experiments/ and return the markdown path."""
    out_dir = out_dir or Path("evals/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"judge-audit_{report.timestamp}_{report.commit}"
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(render_markdown(report))
    payload = {
        "judge_model": report.judge_model,
        "commit": report.commit,
        "timestamp": report.timestamp,
        "n_cases": report.n_cases,
        "prompt_hashes": report.prompt_hashes,
        "base_mean": report.base_mean,
        "shifts": {metric: [vars(s) for s in stats] for metric, stats in report.shifts.items()},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    return md_path


def run_audit_sync(
    cases: list[GoldenCase] | None = None,
    metrics: tuple[str, ...] = ("correctness", "answer_relevancy"),
) -> AuditReport:
    """Sync facade for the CLI."""
    return asyncio.run(run_audit(cases=cases, metrics=metrics))
