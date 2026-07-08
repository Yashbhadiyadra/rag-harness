"""Tests for the golden-set candidate review CLI (ADR-0012, D-a-2).

Fully offline. A ``_ScriptedPrompter`` feeds pre-planned actions to the
review loop and captures everything displayed. No stdin, no $EDITOR,
no ChromaDB, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.expand_golden_set import (
    GoldenCaseCandidate,
    RetrievalEvidence,
    RetrievalHit,
)
from scripts.golden_review import (
    format_candidate,
    load_golden_file,
    load_queue,
    next_id,
    run_review,
    save_queue,
    target_file_for_category,
)

# --- Scripted prompter ------------------------------------------------


class _ScriptedPrompter:
    """A prompter that pops actions and confirmations off a scripted queue.

    - ``actions``: sequence returned by ``prompt_action`` calls.
    - ``confirms``: sequence returned by ``prompt_confirm`` calls.
    - ``editor_output``: value returned by ``edit_text`` regardless of input.
    - ``displayed``: everything the review loop passed to ``display``.
    """

    def __init__(
        self,
        actions: list[str],
        confirms: list[bool] | None = None,
        editor_output: str = "EDITED_ANSWER",
    ) -> None:
        self._actions = list(actions)
        self._confirms = list(confirms or [])
        self._editor_output = editor_output
        self.displayed: list[str] = []
        self.action_calls: list[str] = []
        self.confirm_calls: list[str] = []

    def display(self, text: str) -> None:
        self.displayed.append(text)

    def prompt_action(self, options: str) -> str:
        self.action_calls.append(options)
        return self._actions.pop(0)

    def prompt_confirm(self, question: str) -> bool:
        self.confirm_calls.append(question)
        return self._confirms.pop(0)

    def edit_text(self, initial: str) -> str:
        return self._editor_output


# --- Fixture builders -------------------------------------------------


def _topic_candidate(
    cid: str = "cand-topic-workloads-0000",
    question: str = "What is a Pod?",
    answer: str = "A Pod is the smallest deployable unit.",
) -> GoldenCaseCandidate:
    return GoldenCaseCandidate(
        id=cid,
        category="topic-workloads",
        candidate_question=question,
        source_chunk_id="chunk-abc",
        source_chunk_file="content/en/docs/pod.md",
        source_chunk_text="A Pod is the smallest deployable unit in Kubernetes...",
        suggested_reference_answer=answer,
        suggested_relevant_doc_ids=["content/en/docs/pod.md"],
    )


def _unanswerable_candidate(cid: str = "cand-unanswerable-0000") -> GoldenCaseCandidate:
    return GoldenCaseCandidate(
        id=cid,
        category="unanswerable",
        candidate_question="How do I use K8s 2.0's hyper-mesh feature?",
        source_chunk_id=None,
        source_chunk_file=None,
        source_chunk_text=None,
        suggested_reference_answer=(
            "I do not have enough information in the provided context to answer this question."
        ),
        suggested_relevant_doc_ids=[],
        retrieval_evidence=RetrievalEvidence(
            top_hits=[
                RetrievalHit(
                    chunk_id="chunk-042",
                    source_file="content/en/docs/services.md",
                    similarity=0.42,
                    text_preview="A Service is an abstraction...",
                ),
            ],
            max_similarity=0.42,
            reason=(
                "Top hits are about existing services; none discusses a K8s 2.0 hyper-mesh feature."
            ),
        ),
    )


def _version_sensitive_candidate(
    cid: str = "cand-versionsensitive-0000",
) -> GoldenCaseCandidate:
    return GoldenCaseCandidate(
        id=cid,
        category="version-sensitive",
        candidate_question="Which apiVersion should I use for a Deployment?",
        source_chunk_id="chunk-dep",
        source_chunk_file="content/en/docs/deployment.md",
        source_chunk_text=(
            "The Deployment API graduated to apps/v1 in Kubernetes v1.9. "
            "The apps/v1beta1 and v1beta2 API versions were removed in v1.16."
        ),
        suggested_reference_answer="In v1.32, use apps/v1.",
        suggested_relevant_doc_ids=["content/en/docs/deployment.md"],
    )


def _write_queue(candidates: list[GoldenCaseCandidate], path: Path) -> None:
    save_queue(candidates, path)


# --- Display format ---------------------------------------------------


def test_format_candidate_puts_source_chunk_before_draft_answer_for_topic() -> None:
    """Ground-truth material must appear before LLM suggestion so the
    reviewer verifies down, not trusts up."""
    txt = format_candidate(_topic_candidate(), position=1, total=1)
    chunk_pos = txt.find("SOURCE CHUNK")
    draft_pos = txt.find("DRAFT REFERENCE ANSWER")
    assert chunk_pos != -1 and draft_pos != -1
    assert chunk_pos < draft_pos, "source chunk must appear before draft answer"
    assert "verify" in txt.lower()  # explicit VERIFY language on the chunk box


def test_format_candidate_shows_retrieval_evidence_for_unanswerable() -> None:
    txt = format_candidate(_unanswerable_candidate(), position=1, total=1)
    assert "RETRIEVAL EVIDENCE" in txt
    assert "0.42" in txt  # similarity value
    assert "hyper-mesh" in txt  # LLM reason surfaced


def test_format_candidate_labels_draft_prominently() -> None:
    """The reference answer is ALWAYS labeled as a draft."""
    for cand in (
        _topic_candidate(),
        _unanswerable_candidate(),
        _version_sensitive_candidate(),
    ):
        txt = format_candidate(cand, position=1, total=1)
        assert "DRAFT REFERENCE ANSWER" in txt
        # And the label carries the "verify, do not trust" reminder
        assert "VERIFY" in txt


def test_format_candidate_action_line_lists_all_options() -> None:
    txt = format_candidate(_topic_candidate(), position=1, total=1)
    for opt in ("[y]", "[e]", "[n]", "[f]", "[N]", "[q]"):
        assert opt in txt


# --- Sequential ID assignment -----------------------------------------


def test_next_id_starts_at_001_for_empty_file() -> None:
    assert next_id([], "workloads") == "workloads-001"


def test_next_id_extends_existing_sequence() -> None:
    existing = [{"id": "workloads-001"}, {"id": "workloads-002"}, {"id": "workloads-003"}]
    assert next_id(existing, "workloads") == "workloads-004"


def test_next_id_ignores_wrong_prefix() -> None:
    existing = [{"id": "networking-005"}, {"id": "workloads-002"}]
    assert next_id(existing, "workloads") == "workloads-003"


def test_next_id_handles_hyphenated_prefix() -> None:
    """version-sensitive contains a hyphen; the regex must not split on it."""
    existing = [{"id": "version-sensitive-001"}, {"id": "version-sensitive-002"}]
    assert next_id(existing, "version-sensitive") == "version-sensitive-003"


# --- Target file mapping ----------------------------------------------


def test_target_file_maps_topic_category_to_topic_json(tmp_path: Path) -> None:
    p, prefix = target_file_for_category("topic-workloads", tmp_path)
    assert p == tmp_path / "workloads.json"
    assert prefix == "workloads"


def test_target_file_maps_unanswerable(tmp_path: Path) -> None:
    p, prefix = target_file_for_category("unanswerable", tmp_path)
    assert p == tmp_path / "unanswerable.json"
    assert prefix == "unanswerable"


def test_target_file_maps_version_sensitive(tmp_path: Path) -> None:
    p, prefix = target_file_for_category("version-sensitive", tmp_path)
    assert p == tmp_path / "version-sensitive.json"
    assert prefix == "version-sensitive"


def test_target_file_rejects_unknown_category(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        target_file_for_category("bogus", tmp_path)


# --- run_review actions -----------------------------------------------


def _reload_queue(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_run_review_accept_appends_to_golden_and_marks_accepted(tmp_path: Path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    _write_queue([_topic_candidate()], queue_file)

    prompter = _ScriptedPrompter(actions=["y"])
    summary = run_review(queue_file, golden_dir, prompter)

    assert summary.accepted == 1 and summary.skipped == 0
    # Golden file created and populated
    target = golden_dir / "workloads.json"
    assert target.exists()
    cases = load_golden_file(target)
    assert len(cases) == 1
    assert cases[0]["id"] == "workloads-001"
    assert cases[0]["question"] == "What is a Pod?"
    assert cases[0]["reference_answer"] == "A Pod is the smallest deployable unit."
    # Queue row marked accepted
    rows = _reload_queue(queue_file)
    assert rows[0]["status"] == "accepted"


def test_run_review_edit_uses_editor_output_for_reference_answer(tmp_path: Path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    _write_queue([_topic_candidate()], queue_file)

    prompter = _ScriptedPrompter(actions=["e"], editor_output="  A cleaner answer.  \n")
    summary = run_review(queue_file, golden_dir, prompter)

    assert summary.accepted == 1
    cases = load_golden_file(golden_dir / "workloads.json")
    # Editor output is stripped before storage
    assert cases[0]["reference_answer"] == "A cleaner answer."


def test_run_review_skip_marks_skipped(tmp_path: Path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    _write_queue([_topic_candidate()], queue_file)

    prompter = _ScriptedPrompter(actions=["n"])
    summary = run_review(queue_file, golden_dir, prompter)

    assert summary.skipped == 1 and summary.accepted == 0
    # No golden file created for a purely-skipped session
    assert not (golden_dir / "workloads.json").exists()
    rows = _reload_queue(queue_file)
    assert rows[0]["status"] == "skipped"


def test_run_review_flag_sets_status_flagged(tmp_path: Path) -> None:
    """The [f]lag action must be distinct from skip so ambiguous
    candidates aren't forced into a binary decision under pressure."""
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    _write_queue([_version_sensitive_candidate()], queue_file)

    prompter = _ScriptedPrompter(actions=["f"])
    summary = run_review(queue_file, golden_dir, prompter)

    assert summary.flagged == 1
    rows = _reload_queue(queue_file)
    assert rows[0]["status"] == "flagged"


def test_run_review_bulk_skip_with_confirmation_skips_remaining_in_category(
    tmp_path: Path,
) -> None:
    """[N] must skip only remaining candidates in the SAME category, and
    only after an explicit confirmation."""
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    cands = [
        _topic_candidate(cid="cand-topic-workloads-0000", question="q1"),
        _topic_candidate(cid="cand-topic-workloads-0001", question="q2"),
        _topic_candidate(cid="cand-topic-workloads-0002", question="q3"),
        _unanswerable_candidate(cid="cand-unanswerable-0000"),
    ]
    _write_queue(cands, queue_file)

    # First candidate: bulk-skip, confirmed. Then unanswerable candidate: accept.
    prompter = _ScriptedPrompter(actions=["N", "y"], confirms=[True])
    summary = run_review(queue_file, golden_dir, prompter)

    # All three topic-workloads should be skipped in one action
    assert summary.skipped == 3
    assert summary.accepted == 1
    # Unanswerable file exists
    assert (golden_dir / "unanswerable.json").exists()
    # Confirmation was requested
    assert len(prompter.confirm_calls) == 1
    assert "topic-workloads" in prompter.confirm_calls[0]
    assert "3" in prompter.confirm_calls[0]  # count of remaining in category


def test_run_review_bulk_skip_declined_falls_back_to_single_action(
    tmp_path: Path,
) -> None:
    """When the reviewer fumbles [N] but declines confirmation, they get
    the same candidate back and can pick a real action."""
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    cands = [
        _topic_candidate(cid="cand-topic-workloads-0000"),
        _topic_candidate(cid="cand-topic-workloads-0001"),
    ]
    _write_queue(cands, queue_file)

    # [N] but declines confirmation, then picks [y] on the same candidate,
    # then [n] on the next one
    prompter = _ScriptedPrompter(actions=["N", "y", "n"], confirms=[False])
    summary = run_review(queue_file, golden_dir, prompter)

    assert summary.accepted == 1
    assert summary.skipped == 1


def test_run_review_quit_saves_state_and_leaves_remaining_pending(tmp_path: Path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    cands = [
        _topic_candidate(cid="cand-topic-workloads-0000"),
        _topic_candidate(cid="cand-topic-workloads-0001"),
        _topic_candidate(cid="cand-topic-workloads-0002"),
    ]
    _write_queue(cands, queue_file)

    prompter = _ScriptedPrompter(actions=["y", "q"])
    summary = run_review(queue_file, golden_dir, prompter)

    assert summary.accepted == 1
    assert summary.remaining_pending == 2
    # Queue persisted so the second and third rows are still pending
    rows = _reload_queue(queue_file)
    statuses = [r["status"] for r in rows]
    assert statuses == ["accepted", "pending", "pending"]


def test_run_review_second_pass_skips_already_decided_rows(tmp_path: Path) -> None:
    """A resumed session must not re-present accepted / skipped rows."""
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    cands = [
        _topic_candidate(cid="cand-topic-workloads-0000"),
        _topic_candidate(cid="cand-topic-workloads-0001"),
    ]
    _write_queue(cands, queue_file)

    # First pass: accept then quit
    run_review(queue_file, golden_dir, _ScriptedPrompter(actions=["y", "q"]))

    # Second pass: only the still-pending row should be presented
    prompter2 = _ScriptedPrompter(actions=["n"])
    summary = run_review(queue_file, golden_dir, prompter2)

    assert summary.skipped == 1
    assert summary.accepted == 0
    # Verify only one candidate was displayed on the second pass
    assert len(prompter2.action_calls) == 1


def test_run_review_only_status_flagged_resurfaces_flagged_rows(tmp_path: Path) -> None:
    """After a review session sets aside candidates with [f], the reviewer
    can rerun with only_status='flagged' to work through them."""
    queue_file = tmp_path / "queue.jsonl"
    golden_dir = tmp_path / "golden"
    _write_queue([_topic_candidate()], queue_file)

    # First pass: flag
    run_review(queue_file, golden_dir, _ScriptedPrompter(actions=["f"]))
    rows_after_flag = _reload_queue(queue_file)
    assert rows_after_flag[0]["status"] == "flagged"

    # Second pass with only_status='pending' → nothing to review
    empty = run_review(queue_file, golden_dir, _ScriptedPrompter(actions=[]))
    assert empty.accepted == 0 and empty.skipped == 0 and empty.remaining_pending == 0

    # Third pass with only_status='flagged' → the flagged row is presented
    prompter = _ScriptedPrompter(actions=["y"])
    summary = run_review(queue_file, golden_dir, prompter, only_status="flagged")
    assert summary.accepted == 1


def test_run_review_missing_queue_file_returns_empty_summary(tmp_path: Path) -> None:
    prompter = _ScriptedPrompter(actions=[])
    summary = run_review(tmp_path / "nope.jsonl", tmp_path / "golden", prompter)
    assert summary.accepted == 0
    assert summary.skipped == 0
    assert summary.remaining_pending == 0
    assert prompter.action_calls == []


# --- load_queue graceful failure --------------------------------------


def test_load_queue_skips_malformed_lines(tmp_path: Path) -> None:
    queue_file = tmp_path / "queue.jsonl"
    valid = _topic_candidate()
    with queue_file.open("w", encoding="utf-8") as fh:
        fh.write(valid.model_dump_json() + "\n")
        fh.write("this is not json\n")
        fh.write("\n")  # blank
        fh.write("# comment line\n")
    loaded = load_queue(queue_file)
    assert len(loaded) == 1
    assert loaded[0].id == valid.id
