"""Command-line interface: ingest, query, and eval subcommands."""

import argparse
import sys

from rag_harness.config import settings
from rag_harness.logging_setup import configure_logging

configure_logging(settings.log_level)


def _cmd_ingest(args: argparse.Namespace) -> None:
    from rag_harness.ingest import run_ingest

    run_ingest()


def _cmd_query(args: argparse.Namespace) -> None:
    from rag_harness.generation.corrective import corrective_generate
    from rag_harness.generation.generator import generate
    from rag_harness.retrieval.factory import build_retriever

    retriever = build_retriever(args.strategy)

    if args.corrective:
        result = corrective_generate(args.question, retriever, top_k=args.top_k)
        answer = result.answer
        chunks = result.chunks_used
        print(
            f"\n[corrective: category={result.category.value} "
            f"attempts={result.attempts}"
            + (f" reformulated={result.reformulated_query!r}" if result.reformulated_query else "")
            + "]"
        )
    else:
        chunks = retriever.retrieve(args.question, top_k=args.top_k)
        answer = generate(args.question, chunks)

    print(f"\nAnswer:\n{answer}\n")
    if chunks:
        print("Sources:")
        for c in chunks:
            heading = " > ".join(c.heading_path) if c.heading_path else c.source_file
            print(f"  • {heading}  ({c.source_file})")


def _cmd_eval(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rag_harness.evaluation.runner import export_results, run_eval
    from rag_harness.retrieval.factory import build_retriever

    retriever = build_retriever(args.strategy)
    summary = run_eval(retriever)

    print("\nEvaluation Results")
    print("=" * 56)
    print(f"  Context Recall     : {summary.mean_context_recall:.3f}")
    print(f"  Context Precision  : {summary.mean_context_precision:.3f}")
    print(f"  Faithfulness       : {summary.mean_faithfulness:.3f}")
    print(f"  Correctness        : {summary.mean_correctness:.3f}")
    print(f"  Answer Relevancy   : {summary.mean_answer_relevancy:.3f}")
    print("  " + "-" * 40)
    print(f"  Latency p50        : {summary.latency_p50_ms:>7.0f} ms")
    print(f"  Latency p95        : {summary.latency_p95_ms:>7.0f} ms")
    print(f"  Total cost         : ${summary.total_cost_usd:.4f}")
    print(
        f"  Total tokens       : {summary.total_input_tokens:,} in "
        f"/ {summary.total_output_tokens:,} out"
    )
    print("  " + "-" * 40)
    print(f"  Gate               : {'PASSED' if summary.passed else 'FAILED'}")
    print()

    if args.verbose:
        for result in summary.results:
            print(
                f"[{result.case_id}] recall={result.context_recall:.2f} "
                f"faith={result.faithfulness:.2f} correct={result.correctness:.2f}"
            )
            print(f"  Q: {result.question}")
            print(f"  A: {result.generated_answer[:120]}...")
            print()

    if args.output:
        export_results(summary, Path(args.output))
        print(f"Results written to {args.output}")

    if not summary.passed:
        sys.exit(1)


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = argparse.ArgumentParser(
        prog="rag-harness",
        description="Reliability-first RAG over Kubernetes documentation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Clone, chunk, embed, and index the K8s docs.")

    strategy_choices = ["dense", "hybrid", "hybrid-rerank", "hyde", "full"]

    query_p = sub.add_parser("query", help="Ask a question and print the grounded answer.")
    query_p.add_argument("question", help="The question to answer.")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    query_p.add_argument(
        "--strategy",
        choices=strategy_choices,
        default=settings.retrieval_strategy,
        help="Retrieval strategy (default: %(default)s).",
    )
    query_p.add_argument(
        "--corrective",
        action="store_true",
        default=settings.corrective_rag_enabled,
        help="Enable the corrective critic-and-retry loop (extra LLM calls).",
    )

    eval_p = sub.add_parser("eval", help="Run the golden eval suite and check the quality gate.")
    eval_p.add_argument("--verbose", "-v", action="store_true", help="Print per-case results.")
    eval_p.add_argument(
        "--output",
        metavar="PATH",
        help="Save per-case scores to this file (.json or .csv).",
    )
    eval_p.add_argument(
        "--strategy",
        choices=strategy_choices,
        default=settings.retrieval_strategy,
        help="Retrieval strategy (default: %(default)s).",
    )

    args = parser.parse_args()
    {"ingest": _cmd_ingest, "query": _cmd_query, "eval": _cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
