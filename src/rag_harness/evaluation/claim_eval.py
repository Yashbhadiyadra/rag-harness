"""Claim-level groundedness: classify each atomic claim in an answer (ADR-0027).

Where ``faithfulness_async`` returns one holistic support score, this probe
decomposes the answer into atomic claims and labels each against the retrieved
context with a four-way typology (GSAR-style): grounded, ungrounded,
contradicted, complementary. The per-type breakdown separates a harmless gap
(ungrounded) from a direct conflict with the source (contradicted), which a
single score cannot express.

This is an additive probe. It does not replace the holistic faithfulness judge
or the release gate; see ADR-0027 for the measured comparison and the scope
boundary.
"""

import json
import logging
from dataclasses import dataclass

from rag_harness.evaluation.metrics import _llm_raw_async
from rag_harness.models import Chunk, GoldenCase

logger = logging.getLogger(__name__)

# The four claim types. Kept as plain strings so the renderer and tests compare
# them without an enum; the set is small and fixed by ADR-0027.
GROUNDED = "grounded"
UNGROUNDED = "ungrounded"
CONTRADICTED = "contradicted"
COMPLEMENTARY = "complementary"
VALID_LABELS = frozenset({GROUNDED, UNGROUNDED, CONTRADICTED, COMPLEMENTARY})

_CLAIM_PROMPT = """\
You are an evaluation judge. Given a question, a context (a set of passages), \
and an answer, break the answer into its atomic factual claims and classify \
each claim against the context using exactly one of these labels:

- "grounded": the claim is directly supported by the context.
- "ungrounded": the context does not address the claim (a gap, not a conflict).
- "contradicted": the claim conflicts with what the context says.
- "complementary": the claim is reasonable and does not conflict with the \
context, but goes beyond it (e.g. general background knowledge).

Return ONLY a JSON object of the form:
{"claims": [{"claim": "<text>", "label": "<one label>"}, ...]}
If the answer makes no factual claims, return {"claims": []}.
"""


@dataclass
class ClaimLabel:
    """One atomic claim and its groundedness label."""

    claim: str
    label: str


@dataclass
class ClaimEvalResult:
    """Aggregate claim-type counts over a set of answers."""

    n_cases: int
    n_grounded: int
    n_ungrounded: int
    n_contradicted: int
    n_complementary: int

    @property
    def n_claims(self) -> int:
        return self.n_grounded + self.n_ungrounded + self.n_contradicted + self.n_complementary

    @property
    def groundedness(self) -> float:
        """Strict headline score: fraction of claims directly supported."""
        return self.n_grounded / self.n_claims if self.n_claims else 0.0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of claims that are ungrounded or contradicted (not gaps we allow)."""
        return (self.n_ungrounded + self.n_contradicted) / self.n_claims if self.n_claims else 0.0


def _parse_claims(raw: str) -> list[ClaimLabel]:
    """Parse the judge's JSON reply into labels, degrading safely on garbage.

    A malformed reply yields an empty list (the case contributes no claims)
    rather than raising, so a single bad judge response can never break a run.
    Unknown labels are coerced to ``ungrounded`` - an unverifiable claim must
    not silently count as supported.
    """
    text = raw.strip()
    # Strip a ```json ... ``` fence if the model added one.
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        data = json.loads(text)
        items = data["claims"]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning(
            "claim judge returned unparseable JSON: %r - treating as 0 claims", raw[:120]
        )
        return []
    labels: list[ClaimLabel] = []
    for item in items:
        try:
            claim = str(item["claim"])
            label = str(item["label"]).strip().lower()
        except (KeyError, TypeError):
            continue
        if label not in VALID_LABELS:
            logger.warning("claim judge returned unknown label %r - coercing to ungrounded", label)
            label = UNGROUNDED
        labels.append(ClaimLabel(claim=claim, label=label))
    return labels


async def classify_claims(
    question: str, answer: str, retrieved_chunks: list[Chunk]
) -> list[ClaimLabel]:
    """One structured judge call: extract atomic claims and label each."""
    context = "\n\n".join(c.text for c in retrieved_chunks)
    user_message = f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}"
    raw = await _llm_raw_async(_CLAIM_PROMPT, user_message)
    return _parse_claims(raw)


def aggregate(per_case: list[list[ClaimLabel]]) -> ClaimEvalResult:
    """Roll per-case claim labels into one result."""
    counts = {GROUNDED: 0, UNGROUNDED: 0, CONTRADICTED: 0, COMPLEMENTARY: 0}
    for labels in per_case:
        for cl in labels:
            counts[cl.label] += 1
    return ClaimEvalResult(
        n_cases=len(per_case),
        n_grounded=counts[GROUNDED],
        n_ungrounded=counts[UNGROUNDED],
        n_contradicted=counts[CONTRADICTED],
        n_complementary=counts[COMPLEMENTARY],
    )


async def run_claim_eval(cases: list[GoldenCase], retriever: object) -> ClaimEvalResult:
    """Retrieve, generate, and claim-classify the answer for each golden case."""
    from rag_harness.generation.generator import generate_async

    per_case: list[list[ClaimLabel]] = []
    for case in cases:
        chunks = await retriever.retrieve_async(case.question)  # type: ignore[attr-defined]
        answer = await generate_async(case.question, chunks)
        per_case.append(await classify_claims(case.question, answer, chunks))
    return aggregate(per_case)


def render_markdown(result: ClaimEvalResult, commit: str, timestamp: str) -> str:
    """Render the claim-level groundedness report in the house style."""
    return "\n".join(
        [
            "# Claim-level groundedness (ADR-0027)",
            "",
            f"- Commit: `{commit}`  ·  Timestamp: {timestamp}",
            "- Each answer is split into atomic claims, each labelled against the",
            "  retrieved context. Groundedness is the fraction directly supported;",
            "  contradicted claims are surfaced separately because a conflict is",
            "  worse than a gap.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Cases | {result.n_cases} |",
            f"| Atomic claims | {result.n_claims} |",
            f"| Grounded | {result.n_grounded} |",
            f"| Complementary | {result.n_complementary} |",
            f"| Ungrounded | {result.n_ungrounded} |",
            f"| Contradicted | {result.n_contradicted} |",
            f"| Groundedness | {result.groundedness:.3f} |",
            f"| Hallucination rate | {result.hallucination_rate:.3f} |",
            "",
        ]
    )


def run_claim_eval_sync(cases: list[GoldenCase], strategy: str) -> ClaimEvalResult:
    """Sync facade for the CLI."""
    import asyncio

    from rag_harness.retrieval.factory import build_retriever

    retriever = build_retriever(strategy)
    return asyncio.run(run_claim_eval(cases, retriever))
