"""Load golden cases, run the RAG pipeline per case, and apply the reliability gate."""

import csv
import json
import logging
from pathlib import Path

from rag_harness.config import settings
from rag_harness.evaluation.metrics import context_recall, correctness, faithfulness
from rag_harness.generation.generator import generate
from rag_harness.models import EvalResult, EvalSummary, GoldenCase
from rag_harness.retrieval.base import Retriever

logger = logging.getLogger(__name__)

_GOLDEN_DIR = Path(__file__).parent.parent.parent.parent / "evals" / "golden"


def load_golden_cases(golden_dir: Path | None = None) -> list[GoldenCase]:
    """Load all golden cases from JSON files in the golden directory."""
    directory = golden_dir or _GOLDEN_DIR
    cases: list[GoldenCase] = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        cases.extend(GoldenCase(**item) for item in raw)
    logger.info("loaded %d golden cases from %s", len(cases), directory)
    return cases


def evaluate_case(case: GoldenCase, retriever: Retriever) -> EvalResult:
    """Run the full RAG pipeline for one golden case and score all three metrics."""
    chunks = retriever.retrieve(case.question)
    answer = generate(case.question, chunks)

    recall = context_recall(chunks, case.relevant_doc_ids)
    faith = faithfulness(case.question, answer, chunks)
    correct = correctness(case.question, answer, case.reference_answer)

    logger.debug(
        "case %s — recall=%.2f faithfulness=%.2f correctness=%.2f",
        case.id,
        recall,
        faith,
        correct,
    )
    return EvalResult(
        case_id=case.id,
        question=case.question,
        generated_answer=answer,
        retrieved_doc_ids=[c.source_file for c in chunks],
        context_recall=recall,
        faithfulness=faith,
        correctness=correct,
    )


def run_eval(
    retriever: Retriever,
    golden_dir: Path | None = None,
) -> EvalSummary:
    """Evaluate all golden cases and return a summary with pass/fail gate."""
    cases = load_golden_cases(golden_dir)
    if not cases:
        raise ValueError(f"No golden cases found in {golden_dir or _GOLDEN_DIR}")

    results = [evaluate_case(case, retriever) for case in cases]

    mean_recall = sum(r.context_recall for r in results) / len(results)
    mean_faith = sum(r.faithfulness for r in results) / len(results)
    mean_correct = sum(r.correctness for r in results) / len(results)

    passed = (
        mean_recall >= settings.threshold_context_recall
        and mean_faith >= settings.threshold_faithfulness
        and mean_correct >= settings.threshold_correctness
    )

    summary = EvalSummary(
        results=results,
        mean_context_recall=mean_recall,
        mean_faithfulness=mean_faith,
        mean_correctness=mean_correct,
        passed=passed,
    )

    logger.info(
        "eval complete — recall=%.2f faithfulness=%.2f correctness=%.2f passed=%s",
        mean_recall,
        mean_faith,
        mean_correct,
        passed,
    )
    return summary


def export_results(summary: EvalSummary, output: Path) -> None:
    """Write *summary* to *output* as JSON (.json) or CSV (.csv).

    The file format is inferred from the file extension. JSON preserves the full
    generated answer; CSV is easier to open in a spreadsheet for quick comparison.
    """
    suffix = output.suffix.lower()
    if suffix == ".json":
        output.write_text(
            json.dumps(
                {
                    "mean_context_recall": summary.mean_context_recall,
                    "mean_faithfulness": summary.mean_faithfulness,
                    "mean_correctness": summary.mean_correctness,
                    "passed": summary.passed,
                    "results": [r.model_dump() for r in summary.results],
                },
                indent=2,
            )
        )
    elif suffix == ".csv":
        with output.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "case_id",
                    "question",
                    "context_recall",
                    "faithfulness",
                    "correctness",
                    "generated_answer",
                ],
            )
            writer.writeheader()
            for r in summary.results:
                writer.writerow(
                    {
                        "case_id": r.case_id,
                        "question": r.question,
                        "context_recall": r.context_recall,
                        "faithfulness": r.faithfulness,
                        "correctness": r.correctness,
                        "generated_answer": r.generated_answer,
                    }
                )
    else:
        raise ValueError(f"Unsupported output format: {suffix!r}. Use .json or .csv")
    logger.info("eval results written to %s", output)
