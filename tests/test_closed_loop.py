"""Tests for closed-loop capture of low-confidence queries."""

import json
from pathlib import Path

from rag_harness.evaluation.closed_loop import (
    build_candidate,
    capture_query,
    is_duplicate,
    is_low_confidence,
)
from rag_harness.models import Chunk

_REFUSAL = "I do not have enough information in the provided context to answer this question."


def _chunk(source_file: str, text: str = "some content") -> Chunk:
    return Chunk(
        id=f"{source_file}::0",
        text=text,
        source_file=source_file,
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
    )


def test_is_low_confidence_on_refusal() -> None:
    assert is_low_confidence(_REFUSAL)
    assert not is_low_confidence("A Pod is the smallest deployable unit.")


def test_is_low_confidence_on_low_faithfulness() -> None:
    assert is_low_confidence("A grounded-looking answer.", faithfulness=0.2)
    assert not is_low_confidence("A grounded-looking answer.", faithfulness=0.9)


def test_build_candidate_refusal_is_unanswerable() -> None:
    c = build_candidate("How do I use AWS EKS autoscaling?", _REFUSAL, [])
    assert c["category"] == "unanswerable"
    assert c["suggested_relevant_doc_ids"] == []
    assert c["status"] == "pending"
    assert c["candidate_question"].startswith("How do I use AWS")


def test_build_candidate_answered_infers_topic() -> None:
    chunks = [_chunk("content/en/docs/concepts/services-networking/ingress.md")]
    c = build_candidate("What is an Ingress?", "An Ingress routes traffic.", chunks)
    assert c["category"] == "topic-networking"
    assert c["source_chunk_file"] == "content/en/docs/concepts/services-networking/ingress.md"
    assert c["suggested_relevant_doc_ids"] == [chunks[0].source_file]


def test_build_candidate_matches_review_schema() -> None:
    # the review tool loads candidates via GoldenCaseCandidate; the dict must fit
    from scripts.expand_golden_set import GoldenCaseCandidate

    c = build_candidate("q?", _REFUSAL, [_chunk("content/en/docs/x.md")])
    GoldenCaseCandidate.model_validate(c)  # raises if the shape is wrong


def test_is_duplicate() -> None:
    existing = {"what is an ingress resource"}
    assert is_duplicate("What is an Ingress resource?", existing)
    assert not is_duplicate("How does RBAC work?", existing)


def test_capture_query_writes_low_confidence_and_dedupes(tmp_path: Path) -> None:
    queue = tmp_path / "closed-loop.jsonl"
    chunks = [_chunk("content/en/docs/concepts/storage/pv.md")]

    # a confident answer is not captured
    assert capture_query("What is a PV?", "A PV is storage.", chunks, queue) is False
    assert not queue.exists()

    # a refusal is captured
    assert capture_query("How do I use AWS EKS?", _REFUSAL, [], queue) is True
    assert queue.exists()
    assert len(queue.read_text().splitlines()) == 1

    # the same refusal again is deduped, not re-added
    assert capture_query("How do I use AWS EKS?", _REFUSAL, [], queue) is False
    assert len(queue.read_text().splitlines()) == 1

    # a different low-confidence query is added
    assert capture_query("How do I configure Argo CD?", _REFUSAL, [], queue) is True
    lines = queue.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == "pending"
