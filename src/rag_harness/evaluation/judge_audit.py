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
    # Signed mean of (perturbed - base): direction matters for verbosity
    # bias, where a POSITIVE value means the perturbation inflated scores.
    signed_mean_shift: float = 0.0


def score_shift_stats(
    base: list[float],
    perturbed: list[float],
    threshold: float,
    perturbation: str,
) -> ShiftStats:
    """Summarise how far perturbed scores moved from their base scores.

    ``threshold`` is the reliability-gate cut for this metric; a "flip" is
    a case whose pass/fail verdict changes purely because of formatting.
    ``signed_mean_shift`` preserves direction (perturbed minus base): for
    a verbosity padder, a positive value is the bias - filler raised the
    score without adding correct content.
    """
    if len(base) != len(perturbed) or not base:
        raise ValueError("base and perturbed must be equal-length, non-empty lists")
    deltas = [abs(b - p) for b, p in zip(base, perturbed, strict=True)]
    signed = [p - b for b, p in zip(base, perturbed, strict=True)]
    flips = sum(
        1 for b, p in zip(base, perturbed, strict=True) if (b >= threshold) != (p >= threshold)
    )
    return ShiftStats(
        perturbation=perturbation,
        n=len(base),
        mean_abs_shift=sum(deltas) / len(deltas),
        max_abs_shift=max(deltas),
        flip_rate=flips / len(base),
        signed_mean_shift=sum(signed) / len(signed),
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


# --- Probe answer builders ------------------------------------------------


def truncate_half(text: str) -> str:
    """Keep the first half of the sentences (at least one).

    Deterministic degradation: the result is a genuinely partial answer -
    correct as far as it goes, missing the rest - so its correctness score
    should land near the reliability gate rather than at the ceiling.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    keep = max(1, len(sentences) // 2)
    kept = ". ".join(sentences[:keep])
    return kept if kept.endswith(".") else kept + "."


_VERBOSE_FILLER = (
    " It is worth noting that this is an important topic in this domain, and there are "
    "many considerations involved. Broadly speaking, the details matter and depend on "
    "the specific context and configuration in question. In general, careful attention "
    "to the surrounding factors is advisable when working through this area."
)


def verbose_pad(text: str) -> str:
    """Append fluent, content-free filler - the verbosity-bias probe (JRH).

    The padding adds length and hedging register but zero correct
    information, so a well-behaved correctness judge should not raise its
    score. If it does (positive signed shift), the judge is length-biased:
    it rewards verbosity over substance, the classic LLM-judge failure the
    literature (arXiv:2603.05399) warns about.
    """
    return text.rstrip() + _VERBOSE_FILLER


def cross_pair_answers(cases: list[GoldenCase]) -> list[str]:
    """Answer case i with case i+1's reference (wrapping) - guaranteed wrong.

    The discrimination probe's builder (JRH's "label-flip grading
    accuracy", arXiv:2603.05399): each answer is fluent, on-domain
    Kubernetes prose - but for a different question. A discriminating
    correctness judge must score these low; the fraction it passes at the
    gate is the false-accept rate. Deterministic by construction, and the
    hardest fair negative available without generating text: same corpus,
    same register, same style as real answers.
    """
    if len(cases) < 2:
        raise ValueError("discrimination probe needs at least 2 cases to cross-pair")
    return [cases[(i + 1) % len(cases)].reference_answer for i in range(len(cases))]


# --- Audit runner --------------------------------------------------------


@dataclass
class ProbeResult:
    """Shift statistics for one (probe, metric) pair."""

    probe: str  # "ceiling" | "boundary" | "discrimination"
    metric: str
    base_mean: float  # mean unperturbed score under this probe
    base_pass_rate: float  # fraction of unperturbed scores at/above the gate
    shifts: list[ShiftStats] = field(default_factory=list)


@dataclass
class AuditReport:
    """Full result of one judge-reliability audit run."""

    judge_model: str
    commit: str
    timestamp: str
    n_cases: int
    prompt_hashes: dict[str, str]
    probes: list[ProbeResult] = field(default_factory=list)


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


# Probe -> (answers builder over the case list, metrics it can honestly test).
#
# ceiling: the reference itself - calibration check; scores saturate at
#   1.0, so shifts here understate boundary behavior (ADR-0014 first run).
# boundary: the truncated reference - a partial answer whose correctness
#   sits near the gate, where format-induced verdict flips are possible.
#   Truncation degrades completeness, not topicality, so relevancy is
#   excluded: its boundary scores would still sit at the ceiling and the
#   probe would prove nothing.
# discrimination: another case's reference - fluent, on-domain, wrong.
#   base_pass_rate here IS the false-accept rate; a judge that never
#   fails anything sails through ceiling and boundary but is exposed
#   here. Correctness-only: a cross-paired answer is still topical prose,
#   just for the wrong question, so relevancy is measured incidentally
#   low-signal and excluded.
def _references(cases: list[GoldenCase]) -> list[str]:
    return [c.reference_answer for c in cases]


def _truncated_references(cases: list[GoldenCase]) -> list[str]:
    return [truncate_half(c.reference_answer) for c in cases]


_PROBES: dict[str, tuple[Callable[[list[GoldenCase]], list[str]], tuple[str, ...]]] = {
    "ceiling": (_references, ("correctness", "answer_relevancy")),
    "boundary": (_truncated_references, ("correctness",)),
    "discrimination": (cross_pair_answers, ("correctness",)),
}


async def run_audit(
    cases: list[GoldenCase] | None = None,
    probes: tuple[str, ...] = ("ceiling", "boundary", "discrimination"),
) -> AuditReport:
    """Run the judge audit over the golden set.

    For each probe, build the base answers (reference as-is, degraded, or
    cross-paired wrong), judge them, then judge each format-perturbed
    variant. Perturbations are meaning-preserving, so shifts are pure
    format sensitivity; the discrimination probe additionally reports
    whether formatting can rescue a wrong answer past the gate.
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
    )

    for probe in probes:
        answers_fn, metrics = _PROBES[probe]
        base_answers = answers_fn(cases)
        for metric in metrics:
            threshold = _METRIC_THRESHOLDS[metric]
            base_scores = [
                await _score_case(metric, c, a) for c, a in zip(cases, base_answers, strict=True)
            ]
            result = ProbeResult(
                probe=probe,
                metric=metric,
                base_mean=sum(base_scores) / len(base_scores),
                base_pass_rate=sum(1 for s in base_scores if s >= threshold) / len(base_scores),
            )
            # Verbosity padding runs only where scores have headroom to
            # inflate: the ceiling already sits at 1.0, so a positive shift
            # there is impossible and the probe would be uninformative.
            perturbations = dict(PERTURBATIONS)
            if probe in ("boundary", "discrimination"):
                perturbations["verbose_pad"] = verbose_pad
            for name, perturb in perturbations.items():
                perturbed_scores = [
                    await _score_case(metric, c, perturb(a))
                    for c, a in zip(cases, base_answers, strict=True)
                ]
                result.shifts.append(
                    score_shift_stats(base_scores, perturbed_scores, threshold, name)
                )
                logger.info(
                    "audit %s/%s/%s: mean shift %.3f (signed %+.3f), flips %.0f%%",
                    probe,
                    metric,
                    name,
                    result.shifts[-1].mean_abs_shift,
                    result.shifts[-1].signed_mean_shift,
                    result.shifts[-1].flip_rate * 100,
                )
            report.probes.append(result)
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
        "Probes: **ceiling** judges the reference answer as-is (calibration",
        "check; ~1.0 expected). **boundary** judges the half-truncated",
        "reference - a partial answer scoring near the gate, where",
        "format-induced verdict flips are possible. **discrimination**",
        "judges another case's reference - fluent, on-domain, wrong; its",
        "base pass rate is the judge's false-accept rate and should be ~0%.",
        "Format perturbations are meaning-preserving (shift = format",
        "sensitivity). **verbose_pad** appends content-free filler: a",
        "positive signed shift is verbosity bias - length rewarded over",
        "substance.",
        "",
    ]
    for result in report.probes:
        lines.append(f"## {result.probe} · {result.metric}")
        lines.append("")
        lines.append(
            f"Base mean score: **{result.base_mean:.3f}** · "
            f"base pass rate at gate: **{result.base_pass_rate:.0%}**"
        )
        lines.append("")
        lines.append(
            "| Perturbation | n | Mean abs shift | Signed shift | Max abs shift | Flip rate |"
        )
        lines.append("|---|---|---|---|---|---|")
        for s in result.shifts:
            lines.append(
                f"| {s.perturbation} | {s.n} | {s.mean_abs_shift:.3f} "
                f"| {s.signed_mean_shift:+.3f} | {s.max_abs_shift:.3f} | {s.flip_rate:.0%} |"
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
        "probes": [
            {
                "probe": r.probe,
                "metric": r.metric,
                "base_mean": r.base_mean,
                "base_pass_rate": r.base_pass_rate,
                "shifts": [vars(s) for s in r.shifts],
            }
            for r in report.probes
        ],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    return md_path


def run_audit_sync(
    cases: list[GoldenCase] | None = None,
    probes: tuple[str, ...] = ("ceiling", "boundary", "discrimination"),
) -> AuditReport:
    """Sync facade for the CLI."""
    return asyncio.run(run_audit(cases=cases, probes=probes))
