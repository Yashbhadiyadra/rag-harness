"""Human-labeled judge validation: measure judge-vs-human kappa (ADR-0014 follow-up).

The judge audit reports kappa against *synthetic* ground truth (golden references
are known-correct, cross-paired answers known-incorrect). That is exact by
construction but not the same as agreeing with a human. This harness produces the
credibility anchor the writeup and audit currently lack: a sample a human labels
by hand, scored as chance-corrected agreement between the judge's gate verdict
and the human's verdict.

Flow:
1. ``build_label_sample`` selects a deterministic, balanced sample of
   (question, answer) pairs and writes them to a JSONL file with an empty
   ``human_label`` field.
2. A human fills each ``human_label`` with "correct" or "incorrect".
3. ``score_human_kappa`` runs the correctness judge on each labeled pair and
   reports kappa and raw agreement between judge and human.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_harness.config import settings
from rag_harness.evaluation.judge_audit import cohens_kappa
from rag_harness.evaluation.runner import load_golden_cases

logger = logging.getLogger(__name__)

_VALID_HUMAN_LABELS = frozenset({"correct", "incorrect"})


@dataclass
class LabelItem:
    """One (question, answer) pair for a human to judge correct/incorrect.

    ``reference`` is this question's golden answer (what the judge scores
    against). ``answer`` is the answer under review: for an ``expected="correct"``
    item it is the golden reference itself; for ``expected="incorrect"`` it is a
    different case's reference (fluent, on-domain, wrong). ``expected`` is the
    synthetic label; ``human_label`` is filled in by hand and is the ground truth
    we actually score against.
    """

    id: str
    question: str
    reference: str
    answer: str
    expected: str
    human_label: str = ""


@dataclass
class HumanKappaResult:
    """Judge-vs-human agreement over the hand-labeled sample."""

    n_labeled: int
    kappa: float
    raw_agreement: float
    judge_human_disagreements: int


def build_label_sample(n: int, golden_dir: Path | None = None) -> list[LabelItem]:
    """Build a balanced, deterministic sample of 2*n items to hand-label.

    For each of the first *n* golden cases: one known-correct item (the golden
    reference answers its own question) and one cross-paired item (the next
    case's reference answers this question, so it is on-domain but wrong). The
    human labels every item independently; ``expected`` is not shown as a hint.
    """
    cases = load_golden_cases(golden_dir)
    if n < 1 or len(cases) < 2:
        raise ValueError("need at least 2 golden cases and n >= 1")
    n = min(n, len(cases))
    items: list[LabelItem] = []
    for i in range(n):
        case = cases[i]
        other = cases[(i + 1) % len(cases)]
        items.append(
            LabelItem(
                id=f"{case.id}::correct",
                question=case.question,
                reference=case.reference_answer,
                answer=case.reference_answer,
                expected="correct",
            )
        )
        items.append(
            LabelItem(
                id=f"{case.id}::crosspair",
                question=case.question,
                reference=case.reference_answer,
                answer=other.reference_answer,
                expected="incorrect",
            )
        )
    return items


def write_sample(items: list[LabelItem], path: Path) -> None:
    """Write the sample as JSONL, one item per line, human_label left blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item)) + "\n")


def load_sample(path: Path) -> list[LabelItem]:
    """Load a (possibly hand-labeled) sample from JSONL."""
    items: list[LabelItem] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            items.append(LabelItem(**json.loads(line)))
    return items


async def score_human_kappa(items: list[LabelItem]) -> HumanKappaResult:
    """Score judge-vs-human agreement over the items a human has labeled.

    Items with a blank or invalid ``human_label`` are skipped. For each labeled
    item the correctness judge scores the answer against the question's
    reference; the judge "passes" at or above the correctness threshold. kappa
    is chance-corrected agreement between judge pass/fail and the human verdict.
    """
    from rag_harness.evaluation.metrics import correctness_async

    labeled = [i for i in items if i.human_label.strip().lower() in _VALID_HUMAN_LABELS]
    if not labeled:
        raise ValueError("no items have a valid human_label ('correct' or 'incorrect')")

    threshold = settings.threshold_correctness
    judge_verdicts: list[bool] = []
    human_verdicts: list[bool] = []
    for item in labeled:
        score = await correctness_async(item.question, item.answer, item.reference)
        judge_verdicts.append(score >= threshold)
        human_verdicts.append(item.human_label.strip().lower() == "correct")

    disagreements = sum(1 for j, h in zip(judge_verdicts, human_verdicts, strict=True) if j != h)
    raw = 1.0 - disagreements / len(labeled)
    return HumanKappaResult(
        n_labeled=len(labeled),
        kappa=cohens_kappa(judge_verdicts, human_verdicts),
        raw_agreement=raw,
        judge_human_disagreements=disagreements,
    )


def score_human_kappa_sync(items: list[LabelItem]) -> HumanKappaResult:
    """Sync facade for the CLI."""
    import asyncio

    return asyncio.run(score_human_kappa(items))
