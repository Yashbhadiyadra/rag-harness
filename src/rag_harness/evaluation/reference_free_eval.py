"""Reference-free detection: can we catch wrong answers without a gold label?

The golden set gives every case a reference answer, so correctness (answer
vs reference) is available offline. In production there is no reference -
only the question, the retrieved context, and the answer. Reference-free
hallucination detection (arXiv:2503.21157) asks whether a signal that needs
NO gold label can still flag a wrong answer. Our faithfulness metric is
exactly such a signal: it scores the answer against the retrieved context
alone.

This probe measures how well reference-free faithfulness separates grounded
answers from ungrounded ones. It builds two labeled populations from the
golden set - each case's own reference (grounded in its context) and
another case's reference (not grounded in this context) - scores
faithfulness on both, and reports the separation and a detection accuracy.
A large gap means the reference-free signal is trustworthy enough to gate
production answers where no golden label exists.

Pure helpers (separation stats) are separated from the LLM-calling runner
so they test without a network.
"""

import asyncio
import logging
from dataclasses import dataclass

from rag_harness.evaluation.metrics import faithfulness_async
from rag_harness.models import Chunk, GoldenCase

logger = logging.getLogger(__name__)

# Faithfulness at/above this is treated as "grounded". A wrong answer that
# scores below it is correctly detected without any reference label.
DETECTION_THRESHOLD = 0.5


def _context_chunk(case: GoldenCase) -> list[Chunk]:
    return [
        Chunk(
            id=f"{case.id}::ctx",
            text=case.reference_answer,
            source_file=case.relevant_doc_ids[0] if case.relevant_doc_ids else "synthetic",
            git_commit="synthetic",
            doc_version="synthetic",
            chunk_index=0,
        )
    ]


@dataclass
class ReferenceFreeResult:
    """How well reference-free faithfulness separates grounded from ungrounded."""

    n_cases: int
    mean_faithful_grounded: float  # own reference over own context, want high
    mean_faithful_ungrounded: float  # other reference over this context, want low
    detection_accuracy: float  # fraction of the 2N examples classified correctly

    @property
    def separation(self) -> float:
        """Gap between grounded and ungrounded mean faithfulness."""
        return self.mean_faithful_grounded - self.mean_faithful_ungrounded


def detection_stats(
    grounded: list[float], ungrounded: list[float], threshold: float = DETECTION_THRESHOLD
) -> ReferenceFreeResult:
    """Summarise separation and threshold accuracy for the two populations.

    Grounded answers should score at/above the threshold; ungrounded ones
    below it. Accuracy is over all 2N labeled examples.
    """
    if not grounded or len(grounded) != len(ungrounded):
        raise ValueError("grounded and ungrounded must be equal-length, non-empty lists")
    correct = sum(1 for s in grounded if s >= threshold) + sum(
        1 for s in ungrounded if s < threshold
    )
    total = len(grounded) + len(ungrounded)
    return ReferenceFreeResult(
        n_cases=len(grounded),
        mean_faithful_grounded=sum(grounded) / len(grounded),
        mean_faithful_ungrounded=sum(ungrounded) / len(ungrounded),
        detection_accuracy=correct / total,
    )


async def run_reference_free_probe(cases: list[GoldenCase]) -> ReferenceFreeResult:
    """Score reference-free faithfulness on grounded and ungrounded answers.

    Grounded = case i's reference answered over case i's context. Ungrounded
    = case i+1's reference answered over case i's context (the same
    cross-pairing the judge audit uses for wrong-but-fluent answers).
    """
    if len(cases) < 2:
        raise ValueError("reference-free probe needs at least 2 cases to cross-pair")
    grounded: list[float] = []
    ungrounded: list[float] = []
    for i, case in enumerate(cases):
        context = _context_chunk(case)
        other = cases[(i + 1) % len(cases)]
        grounded.append(await faithfulness_async(case.question, case.reference_answer, context))
        ungrounded.append(await faithfulness_async(case.question, other.reference_answer, context))
    result = detection_stats(grounded, ungrounded)
    logger.info(
        "reference-free: grounded %.3f, ungrounded %.3f, separation %.3f, accuracy %.0f%%",
        result.mean_faithful_grounded,
        result.mean_faithful_ungrounded,
        result.separation,
        result.detection_accuracy * 100,
    )
    return result


def render_markdown(result: ReferenceFreeResult, commit: str, timestamp: str) -> str:
    """Render the reference-free detection report in the house style."""
    return "\n".join(
        [
            "# Reference-free hallucination detection (Phase 2)",
            "",
            f"- Commit: `{commit}`  ·  Timestamp: {timestamp}",
            "- Faithfulness (answer vs context, no gold label) scored on grounded",
            "  answers (own reference) and ungrounded ones (another case's",
            "  reference). A large separation means the reference-free signal can",
            "  gate production answers where no golden reference exists.",
            "",
            "| Population | n | Mean faithfulness |",
            "|---|---|---|",
            f"| Grounded (own reference) | {result.n_cases} "
            f"| {result.mean_faithful_grounded:.3f} |",
            f"| Ungrounded (other reference) | {result.n_cases} "
            f"| {result.mean_faithful_ungrounded:.3f} |",
            "",
            f"Separation: **{result.separation:.3f}** · "
            f"detection accuracy at threshold {DETECTION_THRESHOLD}: "
            f"**{result.detection_accuracy:.0%}**",
            "",
            "Future work: benchmark a dedicated reference-free detector (HHEM,",
            "Lynx, TLM per arXiv:2503.21157) against this baseline. HHEM ships as",
            "an ONNX model; onnxruntime wheels are known to fail in this project's",
            "environment, so it is deferred until that is resolved or a hosted",
            "detector is used.",
            "",
        ]
    )


def run_reference_free_probe_sync(cases: list[GoldenCase]) -> ReferenceFreeResult:
    """Sync facade for the CLI."""
    return asyncio.run(run_reference_free_probe(cases))
