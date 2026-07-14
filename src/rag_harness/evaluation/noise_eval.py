"""Noise robustness: does answer quality survive irrelevant retrieved chunks?

Real retrieval is imperfect - it returns the right passage alongside
plausible-but-irrelevant ones. A robust generator should stay grounded in
the relevant chunk and ignore the noise (the Auepora survey's noise
robustness requirement). This probe holds the correct content fixed and
adds K distractor chunks drawn from other golden cases, then measures how
faithfulness and correctness degrade as K grows.

Pure helpers (context construction, curve stats) are separated from the
LLM-calling runner so they test without a network.
"""

import asyncio
import logging
from dataclasses import dataclass

from rag_harness.evaluation.metrics import correctness_async, faithfulness_async
from rag_harness.generation.generator import generate_async
from rag_harness.models import Chunk, GoldenCase

logger = logging.getLogger(__name__)

DEFAULT_NOISE_LEVELS = (0, 2, 5)


def _chunk_from_case(case: GoldenCase, index: int) -> Chunk:
    return Chunk(
        id=f"{case.id}::noise::{index}",
        text=case.reference_answer,
        source_file=case.relevant_doc_ids[0] if case.relevant_doc_ids else "synthetic",
        git_commit="synthetic",
        doc_version="synthetic",
        chunk_index=index,
    )


def build_noisy_context(correct: GoldenCase, distractors: list[GoldenCase], k: int) -> list[Chunk]:
    """Correct chunk plus *k* distractor chunks, correct one placed first.

    Distractors are other cases' content - real Kubernetes prose that does
    not answer this question. If fewer than *k* distractors are available
    they are cycled.
    """
    ctx = [_chunk_from_case(correct, 0)]
    for i in range(k):
        if not distractors:
            break
        ctx.append(_chunk_from_case(distractors[i % len(distractors)], i + 1))
    return ctx


@dataclass
class NoiseLevelResult:
    """Mean faithfulness and correctness at one noise level."""

    k: int
    n_cases: int
    mean_faithfulness: float
    mean_correctness: float


async def run_noise_probe(
    cases: list[GoldenCase], noise_levels: tuple[int, ...] = DEFAULT_NOISE_LEVELS
) -> list[NoiseLevelResult]:
    """Measure faithfulness/correctness at each noise level.

    For each case the distractors are all the OTHER cases, so the noise is
    always real, on-corpus, but irrelevant to the question at hand.
    """
    results: list[NoiseLevelResult] = []
    for k in noise_levels:
        faith_scores: list[float] = []
        correct_scores: list[float] = []
        for i, case in enumerate(cases):
            distractors = [c for j, c in enumerate(cases) if j != i]
            context = build_noisy_context(case, distractors, k)
            answer = await generate_async(case.question, context)
            faith_scores.append(await faithfulness_async(case.question, answer, context))
            correct_scores.append(
                await correctness_async(case.question, answer, case.reference_answer)
            )
        result = NoiseLevelResult(
            k=k,
            n_cases=len(cases),
            mean_faithfulness=sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
            mean_correctness=sum(correct_scores) / len(correct_scores) if correct_scores else 0.0,
        )
        results.append(result)
        logger.info(
            "noise k=%d: faithfulness %.3f, correctness %.3f",
            k,
            result.mean_faithfulness,
            result.mean_correctness,
        )
    return results


def degradation(results: list[NoiseLevelResult]) -> tuple[float, float]:
    """(faithfulness drop, correctness drop) from the lowest to highest noise level."""
    if len(results) < 2:
        return 0.0, 0.0
    first, last = results[0], results[-1]
    return (
        first.mean_faithfulness - last.mean_faithfulness,
        first.mean_correctness - last.mean_correctness,
    )


def render_markdown(results: list[NoiseLevelResult], commit: str, timestamp: str) -> str:
    """Render the noise-robustness report in the experiments-file house style."""
    faith_drop, correct_drop = degradation(results)
    lines = [
        "# Noise robustness (Phase 2)",
        "",
        f"- Commit: `{commit}`  ·  Timestamp: {timestamp}",
        "- The correct chunk is held fixed; k irrelevant chunks from other",
        "  golden cases are added. A robust generator stays grounded in the",
        "  relevant chunk and ignores the noise.",
        "",
        "| Noise chunks (k) | n | Faithfulness | Correctness |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.k} | {r.n_cases} | {r.mean_faithfulness:.3f} | {r.mean_correctness:.3f} |"
        )
    lines.append("")
    if len(results) >= 2:
        # A negative drop means quality improved with noise; report it as zero
        # degradation rather than a confusing double negative.
        lines.append(
            f"Degradation from k={results[0].k} to k={results[-1].k}: "
            f"faithfulness {max(0.0, faith_drop):.3f}, "
            f"correctness {max(0.0, correct_drop):.3f} "
            "(0.000 = no measurable degradation)."
        )
    lines.append("")
    return "\n".join(lines)


def run_noise_probe_sync(cases: list[GoldenCase]) -> list[NoiseLevelResult]:
    """Sync facade for the CLI."""
    return asyncio.run(run_noise_probe(cases))
