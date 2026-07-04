"""Load golden cases, run the RAG pipeline per case, and apply the reliability gate."""

import csv
import json
import logging
import math
import time
from pathlib import Path

from rag_harness.config import settings
from rag_harness.evaluation.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    correctness,
    faithfulness,
)
from rag_harness.generation.generator import generate
from rag_harness.models import EvalResult, EvalSummary, GoldenCase
from rag_harness.observability.usage import collect_usage
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
    """Run the full RAG pipeline for one golden case and score every metric.

    Measures wall-clock latency for the retrieve + generate + score sequence,
    and aggregates token counts and cost from every LLM call made inside via
    the `collect_usage()` collector.
    """
    start = time.perf_counter()

    with collect_usage() as usage_list:
        chunks = retriever.retrieve(case.question)
        answer = generate(case.question, chunks)

        recall = context_recall(chunks, case.relevant_doc_ids)
        precision = context_precision(case.question, chunks, case.reference_answer)
        faith = faithfulness(case.question, answer, chunks)
        correct = correctness(case.question, answer, case.reference_answer)
        relevancy = answer_relevancy(case.question, answer)

    latency_ms = (time.perf_counter() - start) * 1000.0
    input_tokens = sum(u.input_tokens for u in usage_list)
    output_tokens = sum(u.output_tokens for u in usage_list)
    cost_usd = sum(u.estimated_cost_usd for u in usage_list)

    logger.debug(
        "case %s — recall=%.2f prec=%.2f faith=%.2f correct=%.2f rel=%.2f "
        "latency=%.0fms cost=$%.4f",
        case.id,
        recall,
        precision,
        faith,
        correct,
        relevancy,
        latency_ms,
        cost_usd,
    )
    return EvalResult(
        case_id=case.id,
        question=case.question,
        generated_answer=answer,
        retrieved_doc_ids=[c.source_file for c in chunks],
        context_recall=recall,
        context_precision=precision,
        faithfulness=faith,
        correctness=correct,
        answer_relevancy=relevancy,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost_usd,
    )


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile for a small sample, safe for single-element lists.

    The stdlib `statistics.quantiles` requires at least 2 samples and returns
    cut points, not percentiles at arbitrary rank. For our small golden sets
    (30-100 cases) nearest-rank is unambiguous and easier to reason about.
    """
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = math.ceil(pct / 100.0 * len(sorted_values))
    return sorted_values[max(0, rank - 1)]


def run_eval(
    retriever: Retriever,
    golden_dir: Path | None = None,
) -> EvalSummary:
    """Evaluate all golden cases and return a summary with pass/fail gate."""
    cases = load_golden_cases(golden_dir)
    if not cases:
        raise ValueError(f"No golden cases found in {golden_dir or _GOLDEN_DIR}")

    results = [evaluate_case(case, retriever) for case in cases]
    n = len(results)

    mean_recall = sum(r.context_recall for r in results) / n
    mean_precision = sum(r.context_precision for r in results) / n
    mean_faith = sum(r.faithfulness for r in results) / n
    mean_correct = sum(r.correctness for r in results) / n
    mean_relevancy = sum(r.answer_relevancy for r in results) / n

    latencies = [r.latency_ms for r in results]

    passed = (
        mean_recall >= settings.threshold_context_recall
        and mean_faith >= settings.threshold_faithfulness
        and mean_correct >= settings.threshold_correctness
    )

    summary = EvalSummary(
        results=results,
        mean_context_recall=mean_recall,
        mean_context_precision=mean_precision,
        mean_faithfulness=mean_faith,
        mean_correctness=mean_correct,
        mean_answer_relevancy=mean_relevancy,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        total_cost_usd=sum(r.estimated_cost_usd for r in results),
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        passed=passed,
    )

    logger.info(
        "eval complete — recall=%.2f precision=%.2f faith=%.2f correct=%.2f rel=%.2f "
        "p50=%.0fms p95=%.0fms cost=$%.4f passed=%s",
        mean_recall,
        mean_precision,
        mean_faith,
        mean_correct,
        mean_relevancy,
        summary.latency_p50_ms,
        summary.latency_p95_ms,
        summary.total_cost_usd,
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
                    "mean_context_precision": summary.mean_context_precision,
                    "mean_faithfulness": summary.mean_faithfulness,
                    "mean_correctness": summary.mean_correctness,
                    "mean_answer_relevancy": summary.mean_answer_relevancy,
                    "latency_p50_ms": summary.latency_p50_ms,
                    "latency_p95_ms": summary.latency_p95_ms,
                    "total_cost_usd": summary.total_cost_usd,
                    "total_input_tokens": summary.total_input_tokens,
                    "total_output_tokens": summary.total_output_tokens,
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
