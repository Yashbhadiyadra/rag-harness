"""Closed-loop eval: turn low-confidence production queries into review candidates.

When a live ``/query`` produces a low-confidence answer, that is a signal the
golden set is missing coverage. This module captures such queries as
candidate cases and appends them to a review queue - the SAME human-review
pipeline the golden set is built with (`rag_harness golden review --queue ...`).
The human review gate before the golden set is inviolable: nothing here writes
to the golden files, only to the pending-candidate queue.

Confidence signal on the hot path is cheap: a refusal answer (the generator
could not ground an answer) needs no extra LLM call. An optional faithfulness
score can tighten the signal for offline/batch use.

Pure helpers (confidence decision, candidate construction, dedup) are
separated from the file append so they test without I/O.
"""

import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from rag_harness.evaluation.abstention_eval import is_abstention
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

# Near-duplicate questions above this ratio are not re-captured.
DEDUP_THRESHOLD = 0.85

# Best-effort topic slug from a source path, so an accepted candidate routes
# to the right golden file. Mirrors the generator's mapping; the reviewer can
# override on accept.
_TOPIC_MAP: list[tuple[str, str]] = [
    ("services-networking", "networking"),
    ("networking", "networking"),
    ("storage", "storage"),
    ("scheduling", "scheduling"),
    ("workloads", "workloads"),
    ("configuration", "workloads"),
    ("access-authn-authz", "rbac"),
    ("security", "rbac"),
    ("cluster-administration", "cluster"),
    ("administer-cluster", "cluster"),
    ("architecture", "cluster"),
]


def _topic_of(source_file: str) -> str:
    lower = source_file.lower()
    for needle, topic in _TOPIC_MAP:
        if needle in lower:
            return topic
    return "cluster"


def is_low_confidence(
    answer: str, faithfulness: float | None = None, threshold: float = 0.5
) -> bool:
    """True if the answer looks low-confidence and worth human review.

    A refusal is the cheap hot-path signal; a low reference-free faithfulness
    score is an optional stronger signal for offline capture.
    """
    if is_abstention(answer):
        return True
    return faithfulness is not None and faithfulness < threshold


def _normalize(question: str) -> str:
    return re.sub(r"\W+", " ", question.lower()).strip()


def _candidate_id(question: str) -> str:
    """Deterministic id from the question so the same query is not re-added."""
    return "cand-cl-" + hashlib.sha256(_normalize(question).encode()).hexdigest()[:10]


def build_candidate(question: str, answer: str, chunks: list[Chunk]) -> dict[str, object]:
    """Build a review-queue candidate dict from a query trace.

    A refusal is filed as ``unanswerable`` (the system could not answer);
    otherwise the topic is inferred from the top retrieved chunk. Matches the
    GoldenCaseCandidate schema so `golden review` can load it directly.
    """
    refused = is_abstention(answer)
    top = chunks[0] if chunks else None
    category = "unanswerable" if (refused or top is None) else f"topic-{_topic_of(top.source_file)}"
    hits = [
        {
            "chunk_id": c.id,
            "source_file": c.source_file,
            "similarity": 0.0,
            "text_preview": " ".join(c.text.split())[:200],
        }
        for c in chunks[:5]
    ]
    return {
        "id": _candidate_id(question),
        "category": category,
        "candidate_question": question,
        "source_chunk_id": top.id if top else None,
        "source_chunk_file": top.source_file if top else None,
        "source_chunk_text": top.text if top else None,
        "suggested_reference_answer": answer,
        "suggested_relevant_doc_ids": [top.source_file] if (top and not refused) else [],
        "retrieval_evidence": {
            "top_hits": hits,
            "max_similarity": 0.0,
            "reason": "captured from a low-confidence production query (closed loop)",
        },
        "status": "pending",
    }


def _existing_norms(queue_path: Path) -> set[str]:
    if not queue_path.exists():
        return set()
    norms: set[str] = set()
    for line in queue_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            norms.add(_normalize(json.loads(line).get("candidate_question", "")))
        except (json.JSONDecodeError, AttributeError):
            continue
    return norms


def is_duplicate(question: str, existing_norms: set[str]) -> bool:
    """True if *question* nearly matches one already in the queue."""
    nq = _normalize(question)
    return any(SequenceMatcher(None, nq, e).ratio() >= DEDUP_THRESHOLD for e in existing_norms)


def capture_query(question: str, answer: str, chunks: list[Chunk], queue_path: Path) -> bool:
    """Append a candidate if the query is low-confidence and not a duplicate.

    Returns True if a candidate was written. Never touches the golden files.
    """
    if not is_low_confidence(answer):
        return False
    if is_duplicate(question, _existing_norms(queue_path)):
        return False
    candidate = build_candidate(question, answer, chunks)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a") as fh:
        fh.write(json.dumps(candidate) + "\n")
    logger.info("closed loop: captured candidate %s for review", candidate["id"])
    return True
