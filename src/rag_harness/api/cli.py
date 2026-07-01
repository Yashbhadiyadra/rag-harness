"""Command-line interface: ingest, query, and eval subcommands."""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _cmd_ingest(args: argparse.Namespace) -> None:
    from rag_harness.ingest import run_ingest

    run_ingest()


def _cmd_query(args: argparse.Namespace) -> None:
    from rag_harness.generation.generator import generate
    from rag_harness.retrieval.dense import DenseRetriever

    retriever = DenseRetriever()
    chunks = retriever.retrieve(args.question, top_k=args.top_k)
    answer = generate(args.question, chunks)

    print(f"\nAnswer:\n{answer}\n")
    if chunks:
        print("Sources:")
        for c in chunks:
            heading = " > ".join(c.heading_path) if c.heading_path else c.source_file
            print(f"  • {heading}  ({c.source_file})")


def _cmd_eval(args: argparse.Namespace) -> None:
    from rag_harness.evaluation.runner import run_eval
    from rag_harness.retrieval.dense import DenseRetriever

    retriever = DenseRetriever()
    summary = run_eval(retriever)

    print("\nEvaluation Results")
    print("=" * 40)
    print(f"  Context Recall : {summary.mean_context_recall:.3f}")
    print(f"  Faithfulness   : {summary.mean_faithfulness:.3f}")
    print(f"  Correctness    : {summary.mean_correctness:.3f}")
    print(f"  Gate           : {'PASSED' if summary.passed else 'FAILED'}")
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

    query_p = sub.add_parser("query", help="Ask a question and print the grounded answer.")
    query_p.add_argument("question", help="The question to answer.")
    query_p.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")

    eval_p = sub.add_parser("eval", help="Run the golden eval suite and check the quality gate.")
    eval_p.add_argument("--verbose", "-v", action="store_true", help="Print per-case results.")

    args = parser.parse_args()
    {"ingest": _cmd_ingest, "query": _cmd_query, "eval": _cmd_eval}[args.command](args)


if __name__ == "__main__":
    main()
