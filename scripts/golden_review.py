"""Interactive review of golden-set expansion candidates (ADR-0012, D-a-2).

Walks the review-queue JSONL produced by ``scripts/expand_golden_set.py``,
one candidate at a time. For each pending candidate:

- The source chunk (topic + version-sensitive) or the retrieval evidence
  (unanswerable) is surfaced prominently so the reviewer verifies against
  ground truth, not against the LLM's suggested answer.
- The reviewer chooses:
    [y] accept as-is
    [e] edit the reference answer in $EDITOR, then accept
    [n] skip
    [f] flag for later research (set-aside, revisit with --only-status flagged)
    [N] bulk-skip all remaining pending candidates in this category
        (requires a confirmation prompt, so it cannot be triggered by a
        fumbled keypress)
    [q] quit, saving queue state
- On accept, the case gets the next ID in the target golden file's
  sequence and is appended to ``evals/golden/<category>.json``.

Testable design: a ``Prompter`` protocol wraps every side-effecting
interaction (display, prompt, edit, confirm). Tests inject a scripted
prompter and assert on the resulting queue state and golden-file
mutations. No stdin, no $EDITOR, no ChromaDB, no network.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from scripts.expand_golden_set import (
    GoldenCaseCandidate,
    RetrievalEvidence,
    RetrievalHit,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE_PATH = REPO_ROOT / "evals" / "review-queue" / "candidates.jsonl"
DEFAULT_GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


# --- Prompter interface ------------------------------------------------


class Prompter(Protocol):
    """Every side-effecting interaction the review loop performs.
    Tests inject a scripted implementation."""

    def display(self, text: str) -> None: ...

    def prompt_action(self, options: str) -> str:
        """Ask for a single-char action from the option string."""

    def prompt_confirm(self, question: str) -> bool:
        """Yes/no confirmation. Returns True only on explicit yes."""

    def edit_text(self, initial: str) -> str:
        """Open an editor on ``initial`` and return the edited content."""


# --- Queue and golden-file I/O -----------------------------------------


def load_queue(path: Path) -> list[GoldenCaseCandidate]:
    """Read the review-queue JSONL. Skips blank and comment lines.
    Malformed lines are dropped with a warning rather than aborting."""
    if not path.exists():
        return []
    out: list[GoldenCaseCandidate] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(GoldenCaseCandidate.model_validate_json(line))
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping malformed queue line %d: %s", i, e)
    return out


def save_queue(candidates: list[GoldenCaseCandidate], path: Path) -> None:
    """Overwrite ``path`` with the canonical JSONL for the candidate list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(c.model_dump_json() + "\n")


def load_golden_file(path: Path) -> list[dict[str, Any]]:
    """Load an existing golden JSON. Missing file returns an empty list."""
    if not path.exists():
        return []
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"{path}: expected top-level JSON array")
    # Cast for type-checker; runtime shape is validated by GoldenCase downstream.
    return list(parsed)


def save_golden_file(cases: list[dict[str, Any]], path: Path) -> None:
    """Write the golden JSON with the existing pretty-print style
    (2-space indent, one case per record)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


# --- Category → file + ID sequencing ----------------------------------


_ID_PATTERN = re.compile(r"^(?P<prefix>[a-z-]+)-(?P<num>\d+)$")


def target_file_for_category(category: str, golden_dir: Path) -> tuple[Path, str]:
    """Return (target_file_path, id_prefix) for a candidate category.

    - ``topic-workloads`` → ``golden_dir/workloads.json``, prefix ``workloads``.
    - ``unanswerable`` → ``golden_dir/unanswerable.json``, prefix ``unanswerable``.
    - ``version-sensitive`` → ``golden_dir/version-sensitive.json``, prefix ``version-sensitive``.
    - ``multihop`` → ``golden_dir/multihop.json``, prefix ``multihop``. These
      cases carry two or more ``relevant_doc_ids`` and exist as their own file
      so the ablation can isolate the multi-hop subset.
    """
    if category.startswith("topic-"):
        topic = category[len("topic-") :]
        return golden_dir / f"{topic}.json", topic
    if category == "unanswerable":
        return golden_dir / "unanswerable.json", "unanswerable"
    if category == "version-sensitive":
        return golden_dir / "version-sensitive.json", "version-sensitive"
    if category == "multihop":
        return golden_dir / "multihop.json", "multihop"
    raise ValueError(f"unknown candidate category: {category!r}")


def next_id(existing_cases: list[dict[str, Any]], prefix: str) -> str:
    """Compute the next sequential ID for ``prefix`` based on the highest
    existing numeric suffix. Empty file → ``prefix-001``."""
    highest = 0
    for case in existing_cases:
        m = _ID_PATTERN.match(str(case.get("id", "")))
        if m and m.group("prefix") == prefix:
            highest = max(highest, int(m.group("num")))
    return f"{prefix}-{highest + 1:03d}"


# --- Display formatting -----------------------------------------------


def _wrap_paragraph(text: str, width: int = 76, indent: str = "    ") -> str:
    """Cheap word-wrap for display (no textwrap import needed).
    Preserves existing newlines."""
    out_lines: list[str] = []
    for line in text.splitlines() or [""]:
        words = line.split()
        if not words:
            out_lines.append(indent.rstrip())
            continue
        cur = indent + words[0]
        for w in words[1:]:
            if len(cur) + 1 + len(w) > width + len(indent):
                out_lines.append(cur)
                cur = indent + w
            else:
                cur += " " + w
        out_lines.append(cur)
    return "\n".join(out_lines)


def _format_evidence(evidence: RetrievalEvidence) -> str:
    lines = [f"  Top {len(evidence.top_hits)} hits (max similarity {evidence.max_similarity:.2f}):"]
    for i, h in enumerate(evidence.top_hits, start=1):
        lines.append(f"    {i}. [{h.similarity:.2f}] {h.source_file}")
        lines.append(f"       {h.text_preview[:120]}")
    lines.append("")
    lines.append("  LLM explanation of why these hits do NOT answer the question:")
    lines.append(_wrap_paragraph(evidence.reason, indent="    "))
    return "\n".join(lines)


def format_candidate(candidate: GoldenCaseCandidate, position: int, total: int) -> str:
    """Render the candidate for the reviewer.

    Design deliberate:
    - Ground-truth material (source chunk OR retrieval evidence) appears
      BEFORE the LLM's suggested answer. The reviewer verifies down, not
      trusts up.
    - Every draft output is labeled "DRAFT" so the reviewer never
      mistakes it for an established answer.
    """
    lines = [
        "",
        "=" * 78,
        f"[{position}/{total}]  category: {candidate.category}   id: {candidate.id}",
        "=" * 78,
        "",
        "  Candidate question:",
        _wrap_paragraph(candidate.candidate_question, indent="    "),
        "",
    ]

    if candidate.retrieval_evidence is not None:
        lines.extend(
            [
                "  ┌─ RETRIEVAL EVIDENCE - verify: is this truly not in the corpus? ─",
                _format_evidence(candidate.retrieval_evidence),
                "  └" + "─" * 74,
                "",
            ]
        )
    elif candidate.source_chunk_text is not None:
        lines.append(
            "  ┌─ SOURCE CHUNK - ground truth. Verify draft answer AGAINST THIS. ─"
        )
        source_file = candidate.source_chunk_file or "(unknown source)"
        lines.append(f"  │  Source: {source_file}")
        lines.append("  │")
        # Chunk text: truncate to keep the screen scannable
        chunk_text = candidate.source_chunk_text
        if len(chunk_text) > 1200:
            chunk_text = chunk_text[:1200] + "\n\n    ... (truncated)"
        lines.append(_wrap_paragraph(chunk_text, indent="  │  "))
        lines.append("  └" + "─" * 74)
        lines.append("")

    lines.extend(
        [
            "  ┌─ DRAFT REFERENCE ANSWER - LLM suggestion. VERIFY, do not trust. ─",
            _wrap_paragraph(candidate.suggested_reference_answer, indent="  │  "),
            "  └" + "─" * 74,
            "",
        ]
    )

    if candidate.suggested_relevant_doc_ids:
        lines.append("  Suggested relevant docs:")
        for did in candidate.suggested_relevant_doc_ids:
            lines.append(f"    - {did}")
        lines.append("")

    lines.append(
        "  [y]es include   [e]dit answer in editor   [n]o skip   "
        "[f]lag for later   [N] skip rest of this category   [q]uit"
    )
    return "\n".join(lines)


# --- Review loop --------------------------------------------------------


@dataclass
class ReviewSummary:
    """Return value from ``run_review`` - useful for tests and for
    the CLI to print a session summary."""

    accepted: int = 0
    skipped: int = 0
    flagged: int = 0
    remaining_pending: int = 0
    files_touched: set[Path] = field(default_factory=set)


def _accept_candidate(
    candidate: GoldenCaseCandidate,
    reference_answer: str,
    golden_dir: Path,
) -> Path:
    """Append the candidate as a new golden case with the next sequential ID.
    Returns the touched file path."""
    target_file, prefix = target_file_for_category(candidate.category, golden_dir)
    existing = load_golden_file(target_file)
    new_id = next_id(existing, prefix)
    relevant = list(candidate.suggested_relevant_doc_ids)
    new_case = {
        "id": new_id,
        "question": candidate.candidate_question,
        "reference_answer": reference_answer,
        "relevant_doc_ids": relevant,
    }
    existing.append(new_case)
    save_golden_file(existing, target_file)
    logger.info("accepted %s → %s (%s)", candidate.id, target_file.name, new_id)
    return target_file


def run_review(
    queue_path: Path,
    golden_dir: Path,
    prompter: Prompter,
    only_status: str = "pending",
) -> ReviewSummary:
    """Walk the queue, present each row of ``status == only_status``,
    apply the reviewer's action to the queue and to the golden-set files.

    Idempotent: rows that already have a terminal status (accepted /
    skipped) are not re-presented. Flagged rows can be revisited by
    running again with ``only_status="flagged"``.
    """
    candidates = load_queue(queue_path)
    to_review = [c for c in candidates if c.status == only_status]
    summary = ReviewSummary(remaining_pending=len(to_review))

    for i, candidate in enumerate(to_review, start=1):
        # Bulk-skip in a prior iteration may have already resolved this
        # candidate - do not re-present it.
        if candidate.status != only_status:
            continue

        prompter.display(format_candidate(candidate, i, len(to_review)))
        action = prompter.prompt_action("[y/e/n/f/N/q]").strip()

        if action == "q":
            break

        if action == "y":
            _accept_candidate(candidate, candidate.suggested_reference_answer, golden_dir)
            candidate.status = "accepted"
            summary.accepted += 1
            summary.files_touched.add(
                target_file_for_category(candidate.category, golden_dir)[0]
            )
        elif action == "e":
            edited = prompter.edit_text(candidate.suggested_reference_answer)
            _accept_candidate(candidate, edited.strip(), golden_dir)
            candidate.status = "accepted"
            summary.accepted += 1
            summary.files_touched.add(
                target_file_for_category(candidate.category, golden_dir)[0]
            )
        elif action == "n":
            candidate.status = "skipped"
            summary.skipped += 1
        elif action == "f":
            candidate.status = "flagged"
            summary.flagged += 1
        elif action == "N":
            same_category_remaining = [
                c for c in to_review[i - 1 :] if c.category == candidate.category
            ]
            n_remaining = len(same_category_remaining)
            confirm = prompter.prompt_confirm(
                f"skip remaining {n_remaining} candidate(s) in '{candidate.category}'?"
            )
            if confirm:
                for c in same_category_remaining:
                    c.status = "skipped"
                    summary.skipped += 1
                # Save now so a crash before the next iteration doesn't lose progress
                save_queue(candidates, queue_path)
                continue
            # Confirmation declined - treat the current candidate as if we
            # showed it fresh and let the reviewer pick a real action.
            prompter.display("  (bulk skip declined - presenting this candidate again)")
            action = prompter.prompt_action("[y/e/n/f/q]").strip()
            # Fall-through: rerun the single-candidate branch below for the new action
            if action == "y":
                _accept_candidate(candidate, candidate.suggested_reference_answer, golden_dir)
                candidate.status = "accepted"
                summary.accepted += 1
            elif action == "e":
                edited = prompter.edit_text(candidate.suggested_reference_answer)
                _accept_candidate(candidate, edited.strip(), golden_dir)
                candidate.status = "accepted"
                summary.accepted += 1
            elif action == "n":
                candidate.status = "skipped"
                summary.skipped += 1
            elif action == "f":
                candidate.status = "flagged"
                summary.flagged += 1
            elif action == "q":
                save_queue(candidates, queue_path)
                summary.remaining_pending = sum(
                    1 for c in candidates if c.status == only_status
                )
                return summary
        else:
            prompter.display(f"  (unrecognised action {action!r}; skipping)")
            candidate.status = "skipped"
            summary.skipped += 1

        # Persist after every decision so a mid-session crash keeps progress.
        save_queue(candidates, queue_path)

    summary.remaining_pending = sum(1 for c in candidates if c.status == only_status)
    return summary


# --- Real prompter implementation --------------------------------------


class RealPrompter:
    """Terminal-backed ``Prompter``. Uses stdin, print, and $EDITOR."""

    def display(self, text: str) -> None:
        print(text)

    def prompt_action(self, options: str) -> str:
        return input(f"  {options} > ").strip().lower()

    def prompt_confirm(self, question: str) -> bool:
        response = input(f"  {question} [y/N] > ").strip().lower()
        return response in ("y", "yes")

    def edit_text(self, initial: str) -> str:
        editor = os.environ.get("EDITOR", "vi")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(initial)
            path = Path(fh.name)
        try:
            subprocess.run([editor, str(path)], check=False)
            return path.read_text(encoding="utf-8")
        finally:
            path.unlink(missing_ok=True)


# --- CLI entry point ---------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively review golden-set expansion candidates and append "
            "accepted cases to evals/golden/<category>.json."
        )
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument(
        "--only-status",
        default="pending",
        choices=["pending", "flagged", "accepted", "skipped"],
        help=(
            "Which queue rows to present. Default 'pending'. Rerun with "
            "'flagged' to revisit set-aside cases."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = run_review(
        queue_path=args.queue,
        golden_dir=args.golden_dir,
        prompter=RealPrompter(),
        only_status=args.only_status,
    )

    print()
    print("=" * 56)
    print("Review session summary")
    print(f"  accepted           : {summary.accepted}")
    print(f"  skipped            : {summary.skipped}")
    print(f"  flagged            : {summary.flagged}")
    print(f"  remaining {args.only_status:<10}: {summary.remaining_pending}")
    if summary.files_touched:
        print("  files touched      :")
        for p in sorted(summary.files_touched):
            print(f"    - {p}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Public re-exports for tests (fixes ruff F401)
__all__ = [
    "GoldenCaseCandidate",
    "Prompter",
    "RealPrompter",
    "RetrievalEvidence",
    "RetrievalHit",
    "ReviewSummary",
    "format_candidate",
    "load_golden_file",
    "load_queue",
    "main",
    "next_id",
    "run_review",
    "save_golden_file",
    "save_queue",
    "target_file_for_category",
]
