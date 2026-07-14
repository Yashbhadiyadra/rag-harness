"""Abstention evaluation: does the system refuse when it should?

Negative rejection (the Auepora survey's term) is a first-class
reliability property: a grounded RAG system must answer when the context
supports an answer AND refuse when it does not. Confident answers to
out-of-corpus questions are the most dangerous failure - they look
authoritative and are wrong.

This probe asks questions that a Kubernetes-docs system cannot answer
(cloud-vendor specifics, unrelated products, invented APIs) over
irrelevant-but-real context, and measures how often the system correctly
abstains instead of forcing an answer. It complements the golden set,
which is all answerable cases; once human-reviewed unanswerable cases land
in the golden set this metric folds into the main eval summary.

Pure helpers (abstention detection, rate stats) are separated from the
LLM-calling runner so they test without a network.
"""

import asyncio
import logging
from dataclasses import dataclass

from rag_harness.generation.generator import generate_async
from rag_harness.models import Chunk, GoldenCase

logger = logging.getLogger(__name__)

# Questions no Kubernetes-documentation corpus can answer. A grounded system
# must refuse these, not improvise from unrelated context.
OUT_OF_CORPUS_QUESTIONS: list[str] = [
    "How do I configure the cold-start timeout for an AWS Lambda function?",
    "What is Datadog's per-host monthly price for APM?",
    "How do I reset a Salesforce administrator password?",
    "What is the maximum message size for an Apache Kafka topic on Confluent Cloud?",
    "How do I enable dark mode in the Slack desktop app?",
    "What is the rate limit of the Stripe PaymentIntents API?",
    "How do I rotate credentials for a HashiCorp Vault Transit engine?",
    "What GPU types does Google Colab Pro offer?",
]

# Phrases the generator uses when it declines to answer. Detection is a
# substring match against the known refusal wording (see generator.py and
# corrective.py); kept deliberately narrow so a real answer that merely
# mentions "information" is not miscounted as an abstention.
_ABSTENTION_MARKERS = (
    "enough information in the provided context",
    "don't have enough information",
    "do not have enough information",
)


def is_abstention(answer: str) -> bool:
    """True if the answer is a refusal to answer from context."""
    low = answer.lower()
    return any(marker in low for marker in _ABSTENTION_MARKERS)


@dataclass
class AbstentionResult:
    """Negative-rejection summary over the out-of-corpus question set."""

    n_questions: int
    n_abstained: int

    @property
    def abstention_rate(self) -> float:
        """Fraction of unanswerable questions the system correctly refused."""
        if self.n_questions == 0:
            return 1.0
        return self.n_abstained / self.n_questions


def _irrelevant_context(case: GoldenCase) -> list[Chunk]:
    """A real-but-irrelevant chunk: content from an unrelated golden case."""
    return [
        Chunk(
            id=f"{case.id}::distractor",
            text=case.reference_answer,
            source_file=case.relevant_doc_ids[0] if case.relevant_doc_ids else "synthetic",
            git_commit="synthetic",
            doc_version="synthetic",
            chunk_index=0,
        )
    ]


async def run_abstention_probe(
    distractor_cases: list[GoldenCase],
    questions: list[str] | None = None,
) -> AbstentionResult:
    """Ask each out-of-corpus question over irrelevant context; count refusals.

    Each question is paired with one distractor case's content (cycled) so
    the context is real Kubernetes prose that simply does not answer the
    question - the exact situation where a weak system hallucinates.
    """
    questions = questions or OUT_OF_CORPUS_QUESTIONS
    if not distractor_cases:
        raise ValueError("abstention probe needs at least one distractor case")
    abstained = 0
    for i, question in enumerate(questions):
        case = distractor_cases[i % len(distractor_cases)]
        answer = await generate_async(question, _irrelevant_context(case))
        if is_abstention(answer):
            abstained += 1
    result = AbstentionResult(n_questions=len(questions), n_abstained=abstained)
    logger.info(
        "abstention: %.0f%% (%d/%d correctly refused)",
        result.abstention_rate * 100,
        abstained,
        len(questions),
    )
    return result


def render_markdown(result: AbstentionResult, commit: str, timestamp: str) -> str:
    """Render the abstention report in the experiments-file house style."""
    return "\n".join(
        [
            "# Abstention / negative rejection (Phase 2)",
            "",
            f"- Commit: `{commit}`  ·  Timestamp: {timestamp}",
            "- Out-of-corpus questions asked over irrelevant Kubernetes context.",
            "  A grounded system must refuse; abstention rate is how often it did.",
            "  100% means it never improvised an answer it could not ground.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Questions | {result.n_questions} |",
            f"| Correctly abstained | {result.n_abstained} |",
            f"| Abstention rate | {result.abstention_rate:.0%} |",
            "",
        ]
    )


def run_abstention_probe_sync(distractor_cases: list[GoldenCase]) -> AbstentionResult:
    """Sync facade for the CLI."""
    return asyncio.run(run_abstention_probe(distractor_cases))
