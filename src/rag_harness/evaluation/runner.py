"""Load golden cases, run the RAG pipeline per case, and apply the reliability gate."""

import csv
import json
import logging
import math
import time
from pathlib import Path

from rag_harness.config import settings
from rag_harness.evaluation.history import record_run
from rag_harness.evaluation.metrics import (
    answer_relevancy_async,
    context_precision_async,
    context_recall,
    correctness_async,
    faithfulness_async,
)
from rag_harness.generation.corrective import CorrectiveResult, corrective_generate_async
from rag_harness.generation.generator import generate_async
from rag_harness.models import Chunk, EvalResult, EvalSummary, GoldenCase
from rag_harness.observability.tracing import traced_span
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


async def evaluate_case(
    case: GoldenCase,
    retriever: Retriever,
    *,
    use_corrective: bool = False,
) -> EvalResult:
    """Run the full RAG pipeline for one golden case and score every metric.

    Measures wall-clock latency for the retrieve + generate + score sequence,
    and aggregates token counts and cost from every LLM call made inside via
    the `collect_usage()` collector.

    When *use_corrective* is True, the answer is produced by
    ``corrective_generate_async`` (critic-scored, may reformulate + retry,
    may refuse) instead of the plain generator. Scoring is identical against
    the answer and chunks the corrective path actually used.
    """
    start = time.perf_counter()

    chunks: list[Chunk]
    answer: str
    corrective_result: CorrectiveResult | None = None

    with (
        traced_span("evaluate_case", case_id=case.id) as case_span,
        collect_usage() as usage_list,
    ):
        if use_corrective:
            with traced_span("corrective_generate"):
                corrective_result = await corrective_generate_async(case.question, retriever)
            chunks = corrective_result.chunks_used
            answer = corrective_result.answer
        else:
            with traced_span("retrieve"):
                chunks = await retriever.retrieve_async(case.question)
            with traced_span("generate", chunk_count=len(chunks)):
                answer = await generate_async(case.question, chunks)

        with traced_span("score"):
            recall = context_recall(chunks, case.relevant_doc_ids)
            precision = await context_precision_async(case.question, chunks, case.reference_answer)
            faith = await faithfulness_async(case.question, answer, chunks)
            correct = await correctness_async(case.question, answer, case.reference_answer)
            relevancy = await answer_relevancy_async(case.question, answer)

        if case_span is not None:
            # Record every headline score as a span attribute so the Phoenix
            # UI can filter/sort cases by metric without opening each one.
            case_span.set_attribute("metric.context_recall", recall)
            case_span.set_attribute("metric.context_precision", precision)
            case_span.set_attribute("metric.faithfulness", faith)
            case_span.set_attribute("metric.correctness", correct)
            case_span.set_attribute("metric.answer_relevancy", relevancy)
            if corrective_result is not None:
                case_span.set_attribute("corrective.category", corrective_result.category.value)
                case_span.set_attribute("corrective.attempts", corrective_result.attempts)

    latency_ms = (time.perf_counter() - start) * 1000.0
    input_tokens = sum(u.input_tokens for u in usage_list)
    output_tokens = sum(u.output_tokens for u in usage_list)
    cost_usd = sum(u.estimated_cost_usd for u in usage_list)

    logger.debug(
        "case %s - recall=%.2f prec=%.2f faith=%.2f correct=%.2f rel=%.2f "
        "latency=%.0fms cost=$%.4f corrective=%s",
        case.id,
        recall,
        precision,
        faith,
        correct,
        relevancy,
        latency_ms,
        cost_usd,
        use_corrective,
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
        corrective_category=(corrective_result.category.value if corrective_result else None),
        corrective_attempts=(corrective_result.attempts if corrective_result else None),
        corrective_reformulated_query=(
            corrective_result.reformulated_query if corrective_result else None
        ),
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


async def run_eval(
    retriever: Retriever,
    golden_dir: Path | None = None,
    *,
    use_corrective: bool = False,
    strategy_label: str = "unknown",
    record_history: bool = True,
    case_filter: list[str] | None = None,
) -> EvalSummary:
    """Evaluate all golden cases and return a summary with pass/fail gate.

    When *use_corrective* is True, every case routes through the corrective
    critic-and-retry loop. Baseline metrics stay comparable - the same set of
    quality judges score whatever the corrective path produced.

    Appends one line to ``evals/history/runs.jsonl`` unless *record_history*
    is False. Callers running many configurations (e.g. the ablation runner)
    may prefer to write the history entries themselves with the correct
    strategy label. *strategy_label* is stored verbatim in the history entry.

    When *case_filter* is a non-empty list, only golden cases whose ``id`` is
    in the filter are evaluated. Used by the per-PR CI gate to enforce
    thresholds on a small, cheap subset of the full suite.
    """
    cases = load_golden_cases(golden_dir)
    if case_filter:
        wanted = set(case_filter)
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        raise ValueError(f"No golden cases found in {golden_dir or _GOLDEN_DIR}")

    # Sequential cases (not asyncio.gather) - golden set order matters for
    # reproducibility and the LLM cache benefits from deterministic ordering.
    results = []
    for case in cases:
        result = await evaluate_case(case, retriever, use_corrective=use_corrective)
        results.append(result)
    n = len(results)

    mean_recall = sum(r.context_recall for r in results) / n
    mean_precision = sum(r.context_precision for r in results) / n
    mean_faith = sum(r.faithfulness for r in results) / n
    mean_correct = sum(r.correctness for r in results) / n
    mean_relevancy = sum(r.answer_relevancy for r in results) / n

    # Negative rejection: a case is genuinely unanswerable when its reference
    # answer is itself the refusal. On those, correct behaviour is to refuse,
    # not improvise. Reported as a first-class reliability number.
    from rag_harness.evaluation.abstention_eval import is_abstention

    unanswerable = [
        r for c, r in zip(cases, results, strict=True) if is_abstention(c.reference_answer)
    ]
    n_unanswerable = len(unanswerable)
    abstention_rate = (
        sum(1 for r in unanswerable if is_abstention(r.generated_answer)) / n_unanswerable
        if n_unanswerable
        else 1.0
    )

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
        n_unanswerable=n_unanswerable,
        abstention_rate=abstention_rate,
    )

    logger.info(
        "eval complete - recall=%.2f precision=%.2f faith=%.2f correct=%.2f rel=%.2f "
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
    if record_history:
        record_run(summary, strategy=strategy_label, corrective=use_corrective)
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
                    "n_unanswerable": summary.n_unanswerable,
                    "abstention_rate": summary.abstention_rate,
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
