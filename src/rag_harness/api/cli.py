"""Command-line interface: ingest, query, and eval subcommands."""

import argparse
import asyncio
import sys

from rag_harness.config import settings
from rag_harness.logging_setup import configure_logging

configure_logging(settings.log_level)


def _cmd_ingest(args: argparse.Namespace) -> None:
    from rag_harness.ingest import run_ingest_async

    asyncio.run(run_ingest_async())


def _cmd_query(args: argparse.Namespace) -> None:
    from rag_harness.generation.corrective import corrective_generate_async
    from rag_harness.generation.generator import generate_async
    from rag_harness.retrieval.factory import build_retriever

    retriever = build_retriever(args.strategy)

    from rag_harness.models import Chunk

    async def _run() -> tuple[str, list[Chunk]]:
        if args.corrective:
            r = await corrective_generate_async(args.question, retriever, top_k=args.top_k)
            print(
                f"\n[corrective: category={r.category.value} attempts={r.attempts}"
                + (f" reformulated={r.reformulated_query!r}" if r.reformulated_query else "")
                + "]"
            )
            return r.answer, r.chunks_used
        else:
            c = await retriever.retrieve_async(args.question, top_k=args.top_k)
            a = await generate_async(args.question, c)
            return a, c

    answer, chunks = asyncio.run(_run())

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
    case_filter = None
    if args.subset == "pr":
        case_filter = settings.eval_pr_subset_ids
    elif args.subset:
        case_filter = [s.strip() for s in args.subset.split(",") if s.strip()]

    summary = asyncio.run(
        run_eval(
            retriever,
            use_corrective=args.corrective,
            strategy_label=args.strategy,
            case_filter=case_filter,
        )
    )

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

    if args.corrective:
        # Corrective telemetry - how the critic-and-retry loop actually behaved
        cat_counts: dict[str, int] = {}
        retries = 0
        refusals = 0
        for r in summary.results:
            if r.corrective_category is None:
                continue
            cat_counts[r.corrective_category] = cat_counts.get(r.corrective_category, 0) + 1
            if r.corrective_attempts and r.corrective_attempts > 1:
                retries += 1
            # A refusal returns the "not enough information" answer with no chunks
            if r.corrective_category == "incorrect" and not r.retrieved_doc_ids:
                refusals += 1
        print("Corrective telemetry")
        print("=" * 56)
        parts = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items()))
        print(f"  Categories         : {parts}")
        print(f"  Retries fired      : {retries}")
        print(f"  Refusals           : {refusals}")
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


def _cmd_judge_audit(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rag_harness.evaluation.judge_audit import run_audit_sync, write_report

    # Same reasoning as ablation: the audit re-scores identical inputs
    # across runs, so the judge cache makes reruns effectively free.
    if not args.no_cache:
        settings.llm_cache_enabled = True

    # Judge-model override for the selection matrix (ADR-0014): audits of
    # candidate judges run in their own process, so a process-local settings
    # override is safe - cache keys and usage records both carry the model id.
    if args.judge_model:
        settings.generation_model = args.judge_model

    report = run_audit_sync(retest=args.retest, scales=args.scales)
    md_path = write_report(report, out_dir=Path(args.output_dir))

    print("\nJudge reliability audit (ADR-0014)")
    print("=" * 56)
    print(f"  Judge model  : {report.judge_model}")
    print(f"  Golden cases : {report.n_cases}")
    for result in report.probes:
        print(
            f"\n  {result.probe} · {result.metric} - base mean {result.base_mean:.3f}"
            f", pass rate {result.base_pass_rate:.0%}"
        )
        for s in result.shifts:
            print(
                f"    {s.perturbation:<12} mean shift {s.mean_abs_shift:.3f}"
                f"  signed {s.signed_mean_shift:+.3f}"
                f"  max {s.max_abs_shift:.3f}  flips {s.flip_rate:.0%}"
            )
    for st in report.stability:
        print(
            f"\n  test-retest · {st.metric} (x{st.repeats}) - mean variance "
            f"{st.mean_variance:.4f}, max range {st.max_within_case_range:.3f}, "
            f"{st.n_unstable_cases}/{st.n_cases} unstable"
        )
    for sc in report.scales:
        print(
            f"\n  scale-format · {sc.metric} - mean divergence "
            f"{sc.mean_abs_divergence:.3f} (signed {sc.signed_mean_divergence:+.3f}), "
            f"flips {sc.flip_rate:.0%}"
        )
    print(f"\nReport written to {md_path}")


def _cmd_judge_matrix(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rag_harness.evaluation.judge_audit import run_selection_matrix_sync, write_matrix

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        print("judge-matrix needs at least 2 comma-separated models to compare.")
        sys.exit(1)

    rows, commit, timestamp = run_selection_matrix_sync(models)
    md_path = write_matrix(rows, commit, timestamp, out_dir=Path(args.output_dir))

    print("\nJudge selection matrix (ADR-0014)")
    print("=" * 68)
    print(f"  {'model':<22} {'ceiling':>8} {'worst flip':>11} {'false-acc':>10} {'cost $':>9}")
    for r in rows:
        print(
            f"  {r.model:<22} {r.ceiling_correctness:>8.3f} "
            f"{r.boundary_worst_flip_rate:>10.0%} {r.discrimination_false_accept:>10.0%} "
            f"{r.cost_usd:>9.4f}"
        )
    print(f"\nReport written to {md_path}")


def _cmd_security_eval(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime
    from pathlib import Path

    from rag_harness.evaluation.judge_audit import _current_git_commit
    from rag_harness.evaluation.runner import load_golden_cases
    from rag_harness.evaluation.security_eval import render_markdown, run_poison_probe_sync

    cases = load_golden_cases()
    results = run_poison_probe_sync(cases)

    commit = _current_git_commit()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S+0000")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"poison-resistance_{timestamp}_{commit}.md"
    md_path.write_text(render_markdown(results, commit, timestamp))

    print("\nPoison resistance (Phase 2 security eval)")
    print("=" * 56)
    print(f"  Golden cases : {len(cases)}")
    for r in results:
        print(
            f"  {r.injection:<22} resistance {r.resistance_rate:.0%}"
            f"  ({r.n_compromised}/{r.n_cases} compromised)"
        )
    print(f"\nReport written to {md_path}")


def _cmd_ablation(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rag_harness.evaluation.ablation import (
        relevant_but_incorrect_cases,
        render_csv,
        render_markdown,
        run_ablation,
    )

    # Ablation defaults to opting in to the LLM judge cache - the whole point
    # is to compare 10 configurations against the same golden set cheaply.
    if not args.no_cache:
        settings.llm_cache_enabled = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = asyncio.run(run_ablation())
    if not runs:
        print("No successful ablation configurations. See logs for details.")
        sys.exit(1)

    ts_slug = runs[0].timestamp.replace(":", "").replace("-", "")
    sha = runs[0].git_commit
    md_path = output_dir / f"ablation_{ts_slug}_{sha}.md"
    csv_path = output_dir / f"ablation_{ts_slug}_{sha}.csv"

    md_path.write_text(render_markdown(runs))
    csv_path.write_text(render_csv(runs))

    print("\nAblation results")
    print("=" * 56)
    print(render_markdown(runs))
    print("Relevant-but-incorrect summary")
    print("-" * 40)
    for r in runs:
        rbi = relevant_but_incorrect_cases(r.summary)
        label = f"{r.strategy}" + (" + corrective" if r.corrective else "")
        print(f"  {label:<28} : {len(rbi)} cases ({r.rbi_rate:.0%})")
    print(f"\nMarkdown: {md_path}")
    print(f"CSV:      {csv_path}")


def _cmd_golden(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate `golden <sub>` handler."""
    if args.golden_command == "review":
        # Local import: pulls the reviewer, which pulls the OpenAI SDK etc.
        # Keep the CLI import fast for non-review commands.
        from scripts.golden_review import main as review_main

        argv: list[str] = []
        if args.queue is not None:
            argv += ["--queue", args.queue]
        if args.golden_dir is not None:
            argv += ["--golden-dir", args.golden_dir]
        argv += ["--only-status", args.only_status]
        raise SystemExit(review_main(argv))
    raise ValueError(f"unknown golden subcommand: {args.golden_command!r}")


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
    eval_p.add_argument(
        "--corrective",
        action="store_true",
        default=settings.corrective_rag_enabled,
        help="Route every case through the corrective critic-and-retry loop.",
    )
    eval_p.add_argument(
        "--subset",
        metavar="pr|ID,ID,...",
        help=(
            "Restrict to a subset of golden cases. 'pr' uses "
            "EVAL_PR_SUBSET_IDS (per-PR CI gate). Or pass a comma-separated "
            "list of case IDs."
        ),
    )

    ablation_p = sub.add_parser(
        "ablation",
        help="Run the golden eval across every strategy × mode and emit a comparative table.",
    )
    ablation_p.add_argument(
        "--output-dir",
        default="evals/experiments",
        help="Directory to write ablation_<ts>_<sha>.{md,csv} into (default: %(default)s).",
    )
    ablation_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the LLM judge cache (default: on for ablation runs).",
    )

    judge_audit_p = sub.add_parser(
        "judge-audit",
        help="Audit the LLM judges' format sensitivity against the golden set (ADR-0014).",
    )
    judge_audit_p.add_argument(
        "--output-dir",
        default="evals/experiments",
        help="Directory to write judge-audit_<ts>_<sha>.{md,json} into (default: %(default)s).",
    )
    judge_audit_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the LLM judge cache (default: on for audit runs).",
    )
    judge_audit_p.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help=(
            "Audit a candidate judge model instead of GENERATION_MODEL "
            "(judge selection matrix, ADR-0014). E.g. gpt-4o."
        ),
    )
    judge_audit_p.add_argument(
        "--retest",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run a test-retest stability pass: judge the boundary answers N "
            "times with the cache off and report score variance. N>=2. "
            "Default 0 (skip). Costs N uncached judge calls per case."
        ),
    )
    judge_audit_p.add_argument(
        "--scales",
        action="store_true",
        help=(
            "Run the scale-format sensitivity check: score the boundary "
            "answers on a numeric [0,1] and an A-E letter scale and report "
            "their divergence (ADR-0014)."
        ),
    )

    security_eval_p = sub.add_parser(
        "security-eval",
        help="Measure poison resistance: inject adversarial chunks and check the answer.",
    )
    security_eval_p.add_argument(
        "--output-dir",
        default="evals/experiments",
        help="Directory to write poison-resistance_<ts>_<sha>.md into (default: %(default)s).",
    )

    judge_matrix_p = sub.add_parser(
        "judge-matrix",
        help="Compare candidate judge models on reliability and cost (ADR-0014).",
    )
    judge_matrix_p.add_argument(
        "--models",
        required=True,
        metavar="M1,M2,...",
        help="Comma-separated judge models to compare (at least 2). E.g. gpt-4o-mini,gpt-4o.",
    )
    judge_matrix_p.add_argument(
        "--output-dir",
        default="evals/experiments",
        help="Directory to write judge-matrix_<ts>_<sha>.{md,json} into (default: %(default)s).",
    )

    # Golden-set management: currently one action (review). Uses nested
    # subparsers so future actions (e.g., `golden validate`, `golden stats`)
    # slot in without renaming existing commands.
    golden_p = sub.add_parser("golden", help="Manage the eval golden set.")
    golden_sub = golden_p.add_subparsers(dest="golden_command", required=True)
    review_p = golden_sub.add_parser(
        "review",
        help="Interactively review candidate cases from the expansion queue.",
    )
    review_p.add_argument("--queue", type=str, default=None)
    review_p.add_argument("--golden-dir", type=str, default=None)
    review_p.add_argument(
        "--only-status",
        default="pending",
        choices=["pending", "flagged", "accepted", "skipped"],
    )

    args = parser.parse_args()
    if args.command == "golden":
        _cmd_golden(args)
        return
    {
        "ingest": _cmd_ingest,
        "query": _cmd_query,
        "eval": _cmd_eval,
        "ablation": _cmd_ablation,
        "judge-audit": _cmd_judge_audit,
        "judge-matrix": _cmd_judge_matrix,
        "security-eval": _cmd_security_eval,
    }[args.command](args)


if __name__ == "__main__":
    main()
