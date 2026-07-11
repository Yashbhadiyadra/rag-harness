"""Tests for the golden-set candidate generator (ADR-0012).

Fully offline. Injects fake ChatClient / RetrievalProbe / ChunkSampler
into the pure-function boundary of the generator. No OpenAI, no
ChromaDB, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.expand_golden_set import (
    ChunkSample,
    GoldenCaseCandidate,
    RetrievalHit,
    _balanced_topic_chunks,
    _is_degenerate_answer,
    draft_topic_candidate,
    draft_unanswerable_candidate,
    draft_version_sensitive_candidate,
    generate_candidates,
    write_queue,
)

# --- Fakes ------------------------------------------------------------


class _FakeChatClient:
    """Returns canned JSON payloads. Keyed by system prompt substring to
    let a single fake serve topic / unanswerable / version-sensitive
    prompts in one test."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        # responses: key = substring that appears in the SYSTEM prompt,
        # value = dict returned from chat_completion_json
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def chat_completion_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        for needle, payload in self._responses.items():
            if needle in system:
                return payload
        raise KeyError(f"no fake response registered for system prompt: {system[:60]}")


class _StubProbe:
    """Returns canned retrieval hits regardless of the query."""

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

    def probe(self, question: str, k: int = 5) -> list[RetrievalHit]:
        return self._hits[:k]


class _StubSampler:
    """Returns canned chunks. ``sample_matching`` returns the same chunks
    when the pattern matches any of their text; otherwise empty."""

    def __init__(
        self,
        chunks: list[ChunkSample],
        matching_chunks: list[ChunkSample] | None = None,
    ) -> None:
        self._chunks = chunks
        self._matching = matching_chunks if matching_chunks is not None else []

    def sample(self, n: int, seed: int | None = None) -> list[ChunkSample]:
        return self._chunks[:n]

    def sample_matching(
        self, pattern: re.Pattern[str], n: int, seed: int | None = None
    ) -> list[ChunkSample]:
        return self._matching[:n]


# --- Fixture builders -------------------------------------------------


def _chunk(
    text: str = "A Pod is the smallest deployable unit in Kubernetes.",
    source_file: str = "content/en/docs/concepts/workloads/pods/pod.md",
    heading_path: list[str] | None = None,
) -> ChunkSample:
    return ChunkSample(
        chunk_id="chunk-001",
        source_file=source_file,
        heading_path=heading_path or ["Pods"],
        text=text,
    )


def _hit(similarity: float, source_file: str = "content/en/docs/foo.md") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk-{int(similarity * 1000)}",
        source_file=source_file,
        similarity=similarity,
        text_preview="Some preview text",
    )


# --- _is_degenerate_answer -------------------------------------------


def test_is_degenerate_answer_flags_fence_and_rule_only() -> None:
    # The exact failure observed in the pilot: the draft collapsed to a
    # code-fence marker plus a horizontal rule, no real content.
    assert _is_degenerate_answer("```shell\n---") is True
    assert _is_degenerate_answer("```shell ---") is True
    assert _is_degenerate_answer("```\n```") is True
    assert _is_degenerate_answer("---") is True
    assert _is_degenerate_answer("") is True
    assert _is_degenerate_answer("   \n  ") is True


def test_is_degenerate_answer_keeps_real_answers() -> None:
    # Short but substantive answers must survive.
    assert _is_degenerate_answer("v1") is False
    assert _is_degenerate_answer("tcpEstablishedTimeout: 0s") is False
    # A genuine code-block answer keeps its content after the fence strip.
    assert _is_degenerate_answer("```yaml\napiVersion: v1\n```") is False


# --- draft_topic_candidate -------------------------------------------


def test_draft_topic_candidate_rejects_degenerate_answer() -> None:
    """A draft whose reference answer is only a code fence / rule would
    poison the eval if accepted. The generator must drop it."""
    client = _FakeChatClient(
        {
            "drafts evaluation questions": {
                "question": "What command rolls back a DaemonSet?",
                "reference_answer": "```shell\n---",
            }
        }
    )
    assert draft_topic_candidate(_chunk(), client, candidate_index=0) is None


def test_draft_topic_candidate_success() -> None:
    client = _FakeChatClient(
        {
            "drafts evaluation questions": {
                "question": "What is a Pod?",
                "reference_answer": "A Pod is the smallest deployable unit.",
            }
        }
    )
    cand = draft_topic_candidate(_chunk(), client, candidate_index=0)

    assert cand is not None
    assert cand.category == "topic-workloads"
    assert cand.candidate_question == "What is a Pod?"
    assert cand.suggested_reference_answer == "A Pod is the smallest deployable unit."
    # Source chunk carried through so the reviewer can compare draft answer
    # to the ground-truth chunk text.
    assert cand.source_chunk_text is not None
    assert "smallest deployable unit" in cand.source_chunk_text
    assert cand.source_chunk_file == "content/en/docs/concepts/workloads/pods/pod.md"
    assert cand.suggested_relevant_doc_ids == ["content/en/docs/concepts/workloads/pods/pod.md"]
    assert cand.status == "pending"
    # No retrieval evidence on topic candidates
    assert cand.retrieval_evidence is None


def test_draft_topic_candidate_returns_none_on_malformed_llm_output() -> None:
    """LLM occasionally returns unparseable JSON. Generator must return
    None rather than crash the batch."""
    client = _FakeChatClient(
        {
            "drafts evaluation questions": {
                # Missing 'reference_answer'
                "question": "What is a Pod?",
            }
        }
    )
    assert draft_topic_candidate(_chunk(), client, candidate_index=0) is None


# --- draft_unanswerable_candidate ------------------------------------


def test_draft_unanswerable_candidate_success_below_threshold() -> None:
    """When retrieval evidence shows top hit is below the similarity
    threshold, the candidate is kept, evidence attached, reason recorded."""
    client = _FakeChatClient(
        {
            "draft plausible-sounding": {
                "question": "How do I use K8s 2.0's new hyper-mesh feature?",
                "reference_answer": (
                    "I do not have enough information in the provided context to answer."
                ),
                "why_out_of_corpus": "K8s 2.0 is not in the pinned v1.32 corpus.",
            },
            "explain retrieval evidence": {
                "reason": (
                    "The top hits are all about existing services and networking; "
                    "none discusses a K8s 2.0 hyper-mesh feature."
                )
            },
        }
    )
    probe = _StubProbe(
        [
            _hit(0.42, "content/en/docs/concepts/services-networking/service.md"),
            _hit(0.31, "content/en/docs/concepts/services-networking/ingress.md"),
            _hit(0.28, "content/en/docs/reference/networking/ports-and-protocols.md"),
        ]
    )
    cand = draft_unanswerable_candidate(client, probe, candidate_index=3)

    assert cand is not None
    assert cand.category == "unanswerable"
    assert cand.id == "cand-unanswerable-0003"
    # No source chunk on unanswerables (there IS no correct source chunk)
    assert cand.source_chunk_text is None
    assert cand.source_chunk_id is None
    assert cand.suggested_relevant_doc_ids == []
    # Retrieval evidence attached, with reason text
    assert cand.retrieval_evidence is not None
    assert cand.retrieval_evidence.max_similarity == 0.42
    assert len(cand.retrieval_evidence.top_hits) == 3
    assert "hyper-mesh feature" in cand.retrieval_evidence.reason


def test_draft_unanswerable_candidate_dropped_when_corpus_answers() -> None:
    """When a hit has similarity above threshold, the drafted question
    IS answered by the corpus. Dropping the candidate saves review time."""
    client = _FakeChatClient(
        {
            "draft plausible-sounding": {
                "question": "How do I stop a pod?",  # too close to real corpus
                "reference_answer": "(refusal)",
                "why_out_of_corpus": "(irrelevant, will be dropped)",
            }
        }
    )
    # Top hit at similarity 0.85 is well above the 0.75 drop threshold.
    probe = _StubProbe([_hit(0.85), _hit(0.70)])
    assert draft_unanswerable_candidate(client, probe, candidate_index=0) is None


def test_draft_unanswerable_candidate_falls_back_when_reason_fails() -> None:
    """If the reason-explanation LLM call fails, keep the candidate with
    a placeholder reason instead of dropping it. The reviewer still has
    top hits + question."""
    client = _FakeChatClient(
        {
            "draft plausible-sounding": {
                "question": "How do I use K8s 3.0 whitespace-mode?",
                "reference_answer": "(refusal)",
                "why_out_of_corpus": "K8s 3.0 doesn't exist",
            },
            "explain retrieval evidence": {
                # Missing 'reason' key → parse fails, fallback kicks in
                "wrong_key": "oops",
            },
        }
    )
    probe = _StubProbe([_hit(0.3)])
    cand = draft_unanswerable_candidate(client, probe, candidate_index=0)
    assert cand is not None
    assert cand.retrieval_evidence is not None
    assert "reviewer to inspect" in cand.retrieval_evidence.reason


# --- draft_version_sensitive_candidate -------------------------------


def test_draft_version_sensitive_candidate_success() -> None:
    client = _FakeChatClient(
        {
            "answer differs across K8s versions": {
                "question": "Which apiVersion should I use for a Deployment?",
                "reference_answer": "In v1.32, use apps/v1.",
                "why_version_sensitive": "apps/v1beta1 and v1beta2 were removed in v1.16.",
            }
        }
    )
    chunk = _chunk(
        text=(
            "The Deployment API graduated to apps/v1 in Kubernetes v1.9. "
            "The apps/v1beta1 and v1beta2 API versions were removed in v1.16."
        ),
        source_file="content/en/docs/concepts/workloads/controllers/deployment.md",
    )
    cand = draft_version_sensitive_candidate(chunk, client, candidate_index=2)

    assert cand is not None
    assert cand.category == "version-sensitive"
    assert cand.id == "cand-versionsensitive-0002"
    # Version-sensitive cases MUST carry the chunk prominently so the
    # reviewer verifies the draft against the source, not against the
    # LLM's general knowledge.
    assert cand.source_chunk_text is not None
    assert "apps/v1beta1 and v1beta2 API versions were removed in v1.16" in cand.source_chunk_text


def test_draft_version_sensitive_candidate_rejects_degenerate_answer() -> None:
    client = _FakeChatClient(
        {
            "answer differs across K8s versions": {
                "question": "Which apiVersion should I use?",
                "reference_answer": "```\n```",
                "why_version_sensitive": "n/a",
            }
        }
    )
    assert draft_version_sensitive_candidate(_chunk(), client, candidate_index=0) is None


# --- _balanced_topic_chunks ------------------------------------------


def test_balanced_topic_chunks_spreads_across_topics() -> None:
    """A pool dominated by one topic must not yield an all-one-topic
    selection; thin topics get represented."""
    pool = (
        [_chunk(source_file="content/en/docs/tasks/administer-cluster/x.md") for _ in range(10)]
        + [_chunk(source_file="content/en/docs/reference/access-authn-authz/rbac.md")]
        + [_chunk(source_file="content/en/docs/concepts/storage/volumes.md")]
    )
    picked = _balanced_topic_chunks(pool, n=3, seed=0)
    from scripts.expand_golden_set import _topic_of

    topics = {_topic_of(c.source_file) for c in picked}
    assert len(picked) == 3
    # rbac and storage each have exactly one chunk; round-robin must reach
    # them before taking a third cluster chunk.
    assert "rbac" in topics
    assert "storage" in topics


# --- generate_candidates orchestration -------------------------------


def test_generate_candidates_composes_all_three_categories() -> None:
    client = _FakeChatClient(
        {
            "drafts evaluation questions": {
                "question": "topic Q?",
                "reference_answer": "topic A.",
            },
            "answer differs across K8s versions": {
                "question": "version Q?",
                "reference_answer": "version A.",
                "why_version_sensitive": "yes",
            },
            "draft plausible-sounding": {
                "question": "unanswerable Q?",
                "reference_answer": "(refusal)",
                "why_out_of_corpus": "not in corpus",
            },
            "explain retrieval evidence": {"reason": "hits are off-topic"},
        }
    )
    sampler = _StubSampler(
        chunks=[_chunk(text="pods stuff", source_file="content/en/docs/workloads/x.md")],
        matching_chunks=[_chunk(text="apiVersion foo", source_file="content/en/docs/y.md")],
    )
    probe = _StubProbe([_hit(0.4)])

    candidates = generate_candidates(
        client=client,
        sampler=sampler,
        probe=probe,
        n_topic=1,
        n_unanswerable=1,
        n_version_sensitive=1,
        seed=0,
    )

    categories = [c.category for c in candidates]
    assert "topic-workloads" in categories
    assert "unanswerable" in categories
    assert "version-sensitive" in categories


def test_generate_candidates_dedupes_near_duplicate_questions() -> None:
    """Two chunks that yield the same drafted question must collapse to a
    single candidate so the reviewer does not see the twin."""
    client = _FakeChatClient(
        {
            "drafts evaluation questions": {
                "question": "What are the new features in Kubernetes v1.33 for EKS integration?",
                "reference_answer": "A real answer with content.",
            },
        }
    )
    sampler = _StubSampler(
        chunks=[
            _chunk(text="a", source_file="content/en/docs/workloads/a.md"),
            _chunk(text="b", source_file="content/en/docs/workloads/b.md"),
        ]
    )
    probe = _StubProbe([_hit(0.4)])

    candidates = generate_candidates(
        client=client,
        sampler=sampler,
        probe=probe,
        n_topic=2,  # both chunks draft the identical question
        n_unanswerable=0,
        n_version_sensitive=0,
    )
    assert len(candidates) == 1


def test_generate_candidates_skips_dropped_unanswerables() -> None:
    """Unanswerables that retrieval identifies as in-corpus (similarity
    above threshold) must not appear in the queue."""
    client = _FakeChatClient(
        {
            "draft plausible-sounding": {
                "question": "How do I create a Deployment?",  # in-corpus
                "reference_answer": "(refusal)",
                "why_out_of_corpus": "(placeholder)",
            },
        }
    )
    sampler = _StubSampler(chunks=[], matching_chunks=[])
    probe = _StubProbe([_hit(0.90)])  # very high similarity

    candidates = generate_candidates(
        client=client,
        sampler=sampler,
        probe=probe,
        n_topic=0,
        n_unanswerable=3,  # 3 attempts, all should drop
        n_version_sensitive=0,
    )
    assert candidates == []


# --- write_queue ------------------------------------------------------


def test_write_queue_creates_parent_dir_and_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "queue.jsonl"
    candidates = [
        GoldenCaseCandidate(
            id="cand-topic-workloads-0000",
            category="topic-workloads",
            candidate_question="What is a Pod?",
            source_chunk_id="chunk-001",
            source_chunk_file="content/en/docs/pod.md",
            source_chunk_text="text",
            suggested_reference_answer="answer",
            suggested_relevant_doc_ids=["content/en/docs/pod.md"],
        ),
    ]
    write_queue(candidates, out)

    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["id"] == "cand-topic-workloads-0000"
    assert parsed["status"] == "pending"
    assert parsed["retrieval_evidence"] is None
