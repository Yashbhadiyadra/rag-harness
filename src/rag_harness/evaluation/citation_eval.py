"""Citation accuracy: does each cited passage support the claim it backs?

Chunk-level citations (ADR-0016) are only useful if they are honest - a
citation that points at a passage which does not support the sentence is
worse than no citation, because it manufactures false confidence. This
probe measures two things over the golden set with real retrieval:

- coverage: the fraction of answer sentences that carry a citation.
- accuracy: for each cited (sentence, passage) pair, whether the passage
  actually supports the sentence (judged by the faithfulness scorer with
  that single passage as the only context).

Pure helpers (sentence/marker parsing, aggregate stats) are separated from
the retrieval- and LLM-calling runner so they test without a network.
"""

import asyncio
import logging
import re
from dataclasses import dataclass

from rag_harness.evaluation.metrics import faithfulness_async
from rag_harness.generation.generator import generate_async
from rag_harness.models import Chunk, GoldenCase
from rag_harness.retrieval.base import Retriever

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d+)\]")

# A cited pair counts as accurate when the passage's support score for the
# sentence is at or above this. Matches the reference-free detection cut.
SUPPORT_THRESHOLD = 0.5


def sentences_with_markers(answer: str) -> list[tuple[str, list[int]]]:
    """Split *answer* into sentences, each with the markers it cites.

    Simple ". "/newline segmentation - good enough for the short, factual
    answers this system produces. Each returned marker list is unique and
    sorted; sentences with no citation carry an empty list.
    """
    raw = re.split(r"(?<=[.!?])\s+", answer.strip())
    out: list[tuple[str, list[int]]] = []
    for sentence in raw:
        s = sentence.strip()
        if not s:
            continue
        markers = sorted({int(m.group(1)) for m in _MARKER.finditer(s)})
        out.append((s, markers))
    return out


@dataclass
class CitationEvalResult:
    """Coverage and accuracy of inline citations over the golden set."""

    n_cases: int
    n_sentences: int
    n_cited_sentences: int
    n_citations: int
    n_supported: int

    @property
    def coverage(self) -> float:
        """Fraction of sentences that carry at least one citation."""
        if self.n_sentences == 0:
            return 0.0
        return self.n_cited_sentences / self.n_sentences

    @property
    def accuracy(self) -> float:
        """Fraction of (sentence, cited passage) pairs the passage supports."""
        if self.n_citations == 0:
            return 0.0
        return self.n_supported / self.n_citations


def aggregate(
    per_case: list[tuple[int, int, int, int]],
) -> CitationEvalResult:
    """Fold per-case (sentences, cited_sentences, citations, supported) tuples."""
    n_sentences = sum(t[0] for t in per_case)
    n_cited = sum(t[1] for t in per_case)
    n_citations = sum(t[2] for t in per_case)
    n_supported = sum(t[3] for t in per_case)
    return CitationEvalResult(
        n_cases=len(per_case),
        n_sentences=n_sentences,
        n_cited_sentences=n_cited,
        n_citations=n_citations,
        n_supported=n_supported,
    )


async def run_citation_eval(
    cases: list[GoldenCase], retriever: Retriever, top_k: int = 5
) -> CitationEvalResult:
    """Generate a cited answer per case and score each citation's support."""
    per_case: list[tuple[int, int, int, int]] = []
    for case in cases:
        chunks: list[Chunk] = await retriever.retrieve_async(case.question, top_k=top_k)
        answer = await generate_async(case.question, chunks)
        parsed = sentences_with_markers(answer)

        n_sentences = len(parsed)
        n_cited = sum(1 for _, markers in parsed if markers)
        n_citations = 0
        n_supported = 0
        for sentence, markers in parsed:
            for marker in markers:
                if not (1 <= marker <= len(chunks)):
                    continue  # stray marker, not a real citation
                n_citations += 1
                support = await faithfulness_async(case.question, sentence, [chunks[marker - 1]])
                if support >= SUPPORT_THRESHOLD:
                    n_supported += 1
        per_case.append((n_sentences, n_cited, n_citations, n_supported))

    result = aggregate(per_case)
    logger.info(
        "citation eval: coverage %.0f%%, accuracy %.0f%% (%d/%d citations supported)",
        result.coverage * 100,
        result.accuracy * 100,
        result.n_supported,
        result.n_citations,
    )
    return result


def render_markdown(result: CitationEvalResult, commit: str, timestamp: str) -> str:
    """Render the citation-accuracy report in the house style."""
    return "\n".join(
        [
            "# Citation accuracy (ADR-0016, Phase 3)",
            "",
            f"- Commit: `{commit}`  ·  Timestamp: {timestamp}",
            "- Coverage: answer sentences that carry a citation. Accuracy: cited",
            "  passages that actually support the sentence citing them (judged",
            "  with that single passage as the only context). A high accuracy",
            "  means the citations are honest, not decorative.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Cases | {result.n_cases} |",
            f"| Sentences | {result.n_sentences} |",
            f"| Cited sentences | {result.n_cited_sentences} |",
            f"| Citations | {result.n_citations} |",
            f"| Supported citations | {result.n_supported} |",
            f"| Coverage | {result.coverage:.0%} |",
            f"| Accuracy | {result.accuracy:.0%} |",
            "",
        ]
    )


def run_citation_eval_sync(cases: list[GoldenCase], strategy: str) -> CitationEvalResult:
    """Sync facade for the CLI."""
    from rag_harness.retrieval.factory import build_retriever

    retriever = build_retriever(strategy)
    return asyncio.run(run_citation_eval(cases, retriever))
