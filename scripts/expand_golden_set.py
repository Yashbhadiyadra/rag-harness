# ruff: noqa: E501
"""Generate candidate cases for the eval golden-set expansion (ADR-0012).

Reads the ingested ChromaDB collection, samples chunks, and prompts
``gpt-4o-mini`` to draft candidates in three categories:

- **topic**: grows the existing per-topic files. Each chunk yields one
  question that chunk directly answers, with a draft reference answer
  written from the chunk content only.
- **unanswerable**: deliberately drafts K8s-adjacent questions that the
  pinned corpus does NOT answer, then runs them through the actual
  retriever. Candidates whose top hit is too similar to the drafted
  question get dropped: those are questions the corpus DOES answer.
  Surviving candidates carry ``retrieval_evidence`` so the human
  reviewer can confirm the classification, not trust the classifier.
- **version-sensitive**: chunks that mention specific API versions or
  version-dependent behavior yield questions whose correct answer differs
  across K8s versions. The suggested reference is derived from the chunk
  and the reviewer verifies it against the chunk (not the LLM's draft),
  because the LLM may answer from a newer K8s version than our pin.

Output is a JSONL review queue at ``evals/review-queue/candidates.jsonl``.
No golden-set file is touched here. The ``golden review`` CLI (D-a-2)
walks the queue.

Testable design: the OpenAI client, retriever, and chunk sampler are
injected. Tests provide fakes; no OpenAI calls, no ChromaDB, no network.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE_PATH = REPO_ROOT / "evals" / "review-queue" / "candidates.jsonl"

# Similarity threshold above which we conclude "the corpus does answer
# this" and drop the unanswerable candidate. ChromaDB with cosine
# similarity returns distances in [0, 2]. Converting to similarity via
# ``1 - distance / 2`` gives values in [0, 1]. 0.75 corresponds to a
# solid semantic match; below that the top hit is off-topic.
_UNANSWERABLE_MAX_SIMILARITY = 0.75

# Two normalised questions at or above this character-level ratio are
# treated as near-duplicates; the later candidate is dropped so the
# reviewer never sees the twin.
_DUPLICATE_QUESTION_RATIO = 0.85

# How many chunks to draw into the pool before balancing topic
# candidates across categories. A wide pool lets round-robin selection
# reach thin categories (e.g. rbac) instead of over-sampling whatever
# topic dominates the corpus.
_TOPIC_OVERSAMPLE = 6


# --- Data types ------------------------------------------------------


class ChunkSample(BaseModel):
    """A single chunk drawn from the ingested corpus."""

    chunk_id: str
    source_file: str
    heading_path: list[str] = Field(default_factory=list)
    text: str


class RetrievalHit(BaseModel):
    """One top-k hit for the retrieval-evidence trail on unanswerables."""

    chunk_id: str
    source_file: str
    similarity: float
    text_preview: str


class RetrievalEvidence(BaseModel):
    """The retrieval-evidence bundle attached to unanswerable candidates.

    Written so a human reviewer can confirm the classification directly,
    not just trust that the classifier said "not in corpus."
    """

    top_hits: list[RetrievalHit]
    max_similarity: float
    reason: str  # LLM-written explanation of why the hits don't answer the question


class GoldenCaseCandidate(BaseModel):
    """One drafted candidate. Written to the review queue as a JSONL row."""

    id: str
    category: str  # e.g., "topic-workloads", "unanswerable", "version-sensitive"
    candidate_question: str
    source_chunk_id: str | None = None
    source_chunk_file: str | None = None
    source_chunk_text: str | None = None
    suggested_reference_answer: str
    suggested_relevant_doc_ids: list[str] = Field(default_factory=list)
    retrieval_evidence: RetrievalEvidence | None = None
    status: str = "pending"  # "pending" | "accepted" | "skipped"


# --- Injectable interfaces -------------------------------------------


class ChatClient(Protocol):
    """The subset of the OpenAI client we need. Tests provide a fake."""

    def chat_completion_json(self, system: str, user: str) -> dict[str, Any]: ...


class RetrievalProbe(Protocol):
    """Runs a query through the actual retriever. Returns top-k hits with
    similarity scores. Tests inject a stub that returns canned hits."""

    def probe(self, question: str, k: int = 5) -> list[RetrievalHit]: ...


class ChunkSampler(Protocol):
    """Iterates chunks from the corpus. Tests inject a fixed list."""

    def sample(self, n: int, seed: int | None = None) -> list[ChunkSample]: ...

    def sample_matching(
        self, pattern: re.Pattern[str], n: int, seed: int | None = None
    ) -> list[ChunkSample]: ...


# Type alias used in signatures
type CandidateList = list[GoldenCaseCandidate]


# --- Prompts ----------------------------------------------------------

_TOPIC_SYSTEM = (
    "You are an assistant that drafts evaluation questions for a Kubernetes "
    "documentation RAG system. Every question you draft must be directly "
    "answerable from the specific chunk provided. Do not use any outside "
    "knowledge. Draft a concise reference answer using ONLY the chunk text."
)

_TOPIC_USER_TEMPLATE = (
    "Chunk source: {source_file}\n"
    "Heading path: {heading_path}\n"
    "Chunk text:\n---\n{text}\n---\n\n"
    "Draft ONE realistic question that a Kubernetes user might ask and that "
    "this specific chunk directly answers. Also draft a concise reference "
    "answer written from the chunk content only.\n\n"
    'Return strict JSON: {{"question": "...", "reference_answer": "..."}}. '
    "No prose outside the JSON."
)

_UNANSWERABLE_SYSTEM = (
    "You draft plausible-sounding Kubernetes questions that the current "
    "pinned K8s v1.32 documentation does NOT answer. Focus on: features "
    "introduced after v1.32, cloud-provider-specific integrations (EKS/GKE/"
    "AKS specifics), obscure third-party operators, or features removed "
    "before v1.32. Do not draft questions that clearly overlap core K8s "
    "documentation (Pods, Services, Deployments, RBAC, etc.). Draft the "
    "reference answer as an honest refusal in the style of a RAG system."
)

_UNANSWERABLE_USER = (
    "Draft ONE plausible question that a Kubernetes user might ask and "
    "that the pinned v1.32 documentation does NOT answer.\n\n"
    'Return strict JSON: {"question": "...", "reference_answer": "I do not have enough information in the provided context to answer this question.", "why_out_of_corpus": "..."}. '
    "No prose outside the JSON."
)

_UNANSWERABLE_REASON_TEMPLATE = (
    'The question "{question}" was retrieved against the pinned Kubernetes '
    "corpus. The top {k} hits, in order of similarity, are:\n\n{top_hits}\n\n"
    "In 1-2 short sentences, explain why these hits do NOT actually answer "
    "the question. This explanation is for a human reviewer confirming the "
    'candidate is truly out-of-corpus. Return strict JSON: {{"reason": "..."}}.'
)

_VERSION_SENSITIVE_SYSTEM = (
    "You draft evaluation questions about Kubernetes features whose correct "
    "answer differs across K8s versions. The chunk you are given documents "
    "the behavior for the specific version pinned in this corpus. Draft a "
    "question whose answer would be different in an older or newer K8s. "
    "Also draft a reference answer that is CORRECT for the version this "
    "chunk documents (use the chunk text only, not outside knowledge)."
)

_VERSION_SENSITIVE_USER_TEMPLATE = (
    "Chunk source: {source_file}\n"
    "Heading path: {heading_path}\n"
    "Chunk text:\n---\n{text}\n---\n\n"
    "Draft ONE realistic question about the version-dependent behavior in "
    "this chunk, plus a reference answer derived from the chunk (not from "
    "your general K8s knowledge).\n\n"
    'Return strict JSON: {{"question": "...", "reference_answer": "...", "why_version_sensitive": "..."}}. '
    "No prose outside the JSON."
)

# Regex for chunks that reference specific API versions or version-dependent behavior.
_VERSION_PATTERN = re.compile(
    r"(?:apiVersion|api version|apps/v1beta[12]?|policy/v1beta1|networking\.k8s\.io/v1beta1|"
    r"batch/v1beta1|removed in|deprecated in|since v1\.|graduated in|beta in|alpha in)",
    re.IGNORECASE,
)


# --- Draft validation -------------------------------------------------

# Markdown code-fence lines (```lang / ```), stripped before deciding
# whether a drafted answer has any real content.
_FENCE_LINE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)


def _is_degenerate_answer(answer: str) -> bool:
    """True when a drafted reference answer has no substantive content.

    LLM drafts occasionally collapse to just a Markdown code-fence marker
    plus a horizontal rule (observed in the ADR-0012 pilot: an answer of
    ``` ```shell\\n--- ```). Accepting such a case would poison the eval,
    so we strip fence lines and require at least one alphanumeric
    character to remain. Genuine code-block answers keep their content
    (``apiVersion: v1``) and survive.
    """
    stripped = _FENCE_LINE_RE.sub("", answer)
    return not any(ch.isalnum() for ch in stripped)


# --- Category generators ---------------------------------------------


def draft_topic_candidate(
    chunk: ChunkSample,
    client: ChatClient,
    candidate_index: int,
) -> GoldenCaseCandidate | None:
    """Draft one topic candidate from a chunk. Returns None if the LLM
    output cannot be parsed."""
    topic = _topic_of(chunk.source_file)
    try:
        raw = client.chat_completion_json(
            system=_TOPIC_SYSTEM,
            user=_TOPIC_USER_TEMPLATE.format(
                source_file=chunk.source_file,
                heading_path=" > ".join(chunk.heading_path) or "(no headings)",
                text=chunk.text[:2000],
            ),
        )
        question = str(raw["question"]).strip()
        answer = str(raw["reference_answer"]).strip()
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("topic draft parse failed for chunk %s: %s", chunk.chunk_id, e)
        return None

    if not question or _is_degenerate_answer(answer):
        logger.info("dropping topic draft with empty/degenerate content for chunk %s", chunk.chunk_id)
        return None

    return GoldenCaseCandidate(
        id=f"cand-topic-{topic}-{candidate_index:04d}",
        category=f"topic-{topic}",
        candidate_question=question,
        source_chunk_id=chunk.chunk_id,
        source_chunk_file=chunk.source_file,
        source_chunk_text=chunk.text,
        suggested_reference_answer=answer,
        suggested_relevant_doc_ids=[chunk.source_file],
    )


def draft_unanswerable_candidate(
    client: ChatClient,
    probe: RetrievalProbe,
    candidate_index: int,
) -> GoldenCaseCandidate | None:
    """Draft one unanswerable candidate, verify against retrieval, and
    attach retrieval evidence. Returns None when the drafted question is
    actually answered by the corpus (top hit above the similarity
    threshold) or when either LLM call fails to parse."""
    try:
        drafted = client.chat_completion_json(
            system=_UNANSWERABLE_SYSTEM,
            user=_UNANSWERABLE_USER,
        )
        question = str(drafted["question"]).strip()
        answer = str(drafted["reference_answer"]).strip()
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("unanswerable draft parse failed: %s", e)
        return None

    hits = probe.probe(question, k=5)
    max_sim = max((h.similarity for h in hits), default=0.0)
    if max_sim >= _UNANSWERABLE_MAX_SIMILARITY:
        # Corpus does answer this. Drop the candidate; don't waste review
        # time on it.
        logger.info(
            "dropping unanswerable candidate (max similarity %.2f above threshold): %.80s",
            max_sim,
            question,
        )
        return None

    top_hits_text = "\n".join(
        f"  {i + 1}. [{h.similarity:.2f}] {h.source_file}: {h.text_preview[:120]}"
        for i, h in enumerate(hits)
    )
    try:
        reason_raw = client.chat_completion_json(
            system="You explain retrieval evidence to human reviewers of a RAG eval set. Be terse.",
            user=_UNANSWERABLE_REASON_TEMPLATE.format(
                question=question,
                k=len(hits),
                top_hits=top_hits_text,
            ),
        )
        reason = str(reason_raw["reason"]).strip()
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("unanswerable reason parse failed: %s", e)
        reason = "(LLM did not produce a reason; reviewer to inspect top hits below)"

    evidence = RetrievalEvidence(top_hits=hits, max_similarity=max_sim, reason=reason)

    return GoldenCaseCandidate(
        id=f"cand-unanswerable-{candidate_index:04d}",
        category="unanswerable",
        candidate_question=question,
        source_chunk_id=None,
        source_chunk_file=None,
        source_chunk_text=None,
        suggested_reference_answer=answer,
        suggested_relevant_doc_ids=[],
        retrieval_evidence=evidence,
    )


def draft_version_sensitive_candidate(
    chunk: ChunkSample,
    client: ChatClient,
    candidate_index: int,
) -> GoldenCaseCandidate | None:
    """Draft one version-sensitive candidate from a chunk that references
    a specific API version or version-dependent behavior."""
    try:
        raw = client.chat_completion_json(
            system=_VERSION_SENSITIVE_SYSTEM,
            user=_VERSION_SENSITIVE_USER_TEMPLATE.format(
                source_file=chunk.source_file,
                heading_path=" > ".join(chunk.heading_path) or "(no headings)",
                text=chunk.text[:2000],
            ),
        )
        question = str(raw["question"]).strip()
        answer = str(raw["reference_answer"]).strip()
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(
            "version-sensitive draft parse failed for chunk %s: %s", chunk.chunk_id, e
        )
        return None

    if not question or _is_degenerate_answer(answer):
        logger.info(
            "dropping version-sensitive draft with empty/degenerate content for chunk %s",
            chunk.chunk_id,
        )
        return None

    return GoldenCaseCandidate(
        id=f"cand-versionsensitive-{candidate_index:04d}",
        category="version-sensitive",
        candidate_question=question,
        source_chunk_id=chunk.chunk_id,
        source_chunk_file=chunk.source_file,
        source_chunk_text=chunk.text,
        suggested_reference_answer=answer,
        suggested_relevant_doc_ids=[chunk.source_file],
    )


# --- Helpers ----------------------------------------------------------


def _topic_of(source_file: str) -> str:
    """Derive an existing golden-set topic slug from a source file path.

    Maps ``content/en/docs/tasks/administer-cluster/...`` → ``cluster``
    (a best-effort classification; the reviewer can override on accept).
    """
    lower = source_file.lower()
    mapping = [
        ("networking", "networking"),
        ("services-networking", "networking"),
        ("storage", "storage"),
        ("scheduling", "scheduling"),
        ("scheduling-eviction", "scheduling"),
        ("workloads", "workloads"),
        ("configuration", "workloads"),
        ("access-authn-authz", "rbac"),
        ("security", "rbac"),
        ("cluster-administration", "cluster"),
        ("administer-cluster", "cluster"),
        ("architecture", "cluster"),
    ]
    for needle, topic in mapping:
        if needle in lower:
            return topic
    return "cluster"


def _balanced_topic_chunks(
    pool: list[ChunkSample], n: int, seed: int
) -> list[ChunkSample]:
    """Pick *n* chunks from *pool* spread evenly across topics.

    Global random sampling over-represents whatever topic dominates the
    corpus (in the pilot, 13 of 20 topic drafts came from cluster docs
    while rbac got none). Bucketing by :func:`_topic_of` and selecting
    round-robin across topics gives thin categories a fair share.
    """
    buckets: dict[str, list[ChunkSample]] = defaultdict(list)
    for chunk in pool:
        buckets[_topic_of(chunk.source_file)].append(chunk)

    rng = random.Random(seed)
    order = sorted(buckets)
    for topic in order:
        rng.shuffle(buckets[topic])

    picked: list[ChunkSample] = []
    i = 0
    while len(picked) < n and any(buckets[topic] for topic in order):
        topic = order[i % len(order)]
        if buckets[topic]:
            picked.append(buckets[topic].pop())
        i += 1
    return picked


def _normalize_question(question: str) -> str:
    """Lowercase and collapse to alphanumeric words for duplicate matching."""
    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def _dedupe_candidates(candidates: CandidateList) -> CandidateList:
    """Drop near-duplicate questions, keeping the first occurrence.

    LLM drafting produces paraphrase pairs (the pilot yielded two nearly
    identical "v1.33 features for EKS integration" unanswerables). A
    character-level ratio at or above ``_DUPLICATE_QUESTION_RATIO`` marks
    the later candidate as a twin and removes it from the review queue.
    """
    kept: CandidateList = []
    kept_norms: list[str] = []
    for cand in candidates:
        norm = _normalize_question(cand.candidate_question)
        if any(
            SequenceMatcher(None, norm, prior).ratio() >= _DUPLICATE_QUESTION_RATIO
            for prior in kept_norms
        ):
            logger.info("dropping near-duplicate candidate: %.80s", cand.candidate_question)
            continue
        kept.append(cand)
        kept_norms.append(norm)
    return kept


def write_queue(candidates: CandidateList, path: Path) -> None:
    """Write candidates as JSONL to *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(c.model_dump_json() + "\n")


# --- Real-world adapters (only used from main()) ---------------------


class _OpenAIChatClient:
    """Real ``ChatClient`` backed by the OpenAI SDK. Requires
    ``OPENAI_API_KEY``. Not used in tests."""

    def __init__(self, model: str) -> None:
        self._client = OpenAI()
        self._model = model

    def chat_completion_json(self, system: str, user: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.7,  # slightly warm for question variety
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed: dict[str, Any] = json.loads(content)
        return parsed


class _ChromaChunkSampler:
    """Real ``ChunkSampler`` backed by the ingested ChromaDB collection.
    Not used in tests."""

    def __init__(self) -> None:
        import chromadb  # local import: heavy dep

        from rag_harness.config import settings

        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self._collection = client.get_collection(name=settings.chroma_collection)

    def _load_all(self) -> list[ChunkSample]:
        results = self._collection.get(include=["documents", "metadatas"])
        ids = results.get("ids", []) or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        out: list[ChunkSample] = []
        for cid, doc, meta in zip(ids, docs, metas, strict=True):
            heading_path: list[str] = []
            if isinstance(meta, dict) and isinstance(meta.get("heading_path"), str):
                try:
                    heading_path = json.loads(meta["heading_path"])
                except json.JSONDecodeError:
                    heading_path = []
            out.append(
                ChunkSample(
                    chunk_id=str(cid),
                    source_file=str((meta or {}).get("source_file", "unknown")),
                    heading_path=heading_path,
                    text=str(doc or ""),
                )
            )
        return out

    def sample(self, n: int, seed: int | None = None) -> list[ChunkSample]:
        all_chunks = self._load_all()
        rng = random.Random(seed)
        return rng.sample(all_chunks, min(n, len(all_chunks)))

    def sample_matching(
        self, pattern: re.Pattern[str], n: int, seed: int | None = None
    ) -> list[ChunkSample]:
        matching = [c for c in self._load_all() if pattern.search(c.text)]
        rng = random.Random(seed)
        return rng.sample(matching, min(n, len(matching)))


class _ChromaRetrievalProbe:
    """Real ``RetrievalProbe`` backed by the existing dense retriever.

    Uses a fresh embed for the drafted question and returns hits with a
    computed similarity (1 - distance/2) so downstream evidence rendering
    reads consistently regardless of Chroma's distance metric."""

    def __init__(self) -> None:
        import chromadb  # local import: heavy dep
        from openai import OpenAI as _OpenAI

        from rag_harness.config import settings

        self._openai = _OpenAI()
        self._embedding_model = settings.embedding_model
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self._collection = client.get_collection(name=settings.chroma_collection)

    def probe(self, question: str, k: int = 5) -> list[RetrievalHit]:
        emb_response = self._openai.embeddings.create(
            model=self._embedding_model,
            input=[question],
        )
        vector = emb_response.data[0].embedding
        results = self._collection.query(
            query_embeddings=[vector],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        hits: list[RetrievalHit] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            similarity = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
            hits.append(
                RetrievalHit(
                    chunk_id=str(cid),
                    source_file=str((meta or {}).get("source_file", "unknown")),
                    similarity=similarity,
                    text_preview=str(doc or "")[:400],
                )
            )
        return hits


# --- Main orchestration ----------------------------------------------


def generate_candidates(
    client: ChatClient,
    sampler: ChunkSampler,
    probe: RetrievalProbe,
    n_topic: int,
    n_unanswerable: int,
    n_version_sensitive: int,
    seed: int = 42,
) -> CandidateList:
    """Compose all three category generators into a single candidate list.
    Deterministic given the same seed and the same fakes/inputs.
    """
    candidates: CandidateList = []

    # Topic candidates: draw a wide pool, then balance across topics so
    # thin categories are represented rather than the corpus-dominant one.
    if n_topic > 0:
        pool = sampler.sample(n_topic * _TOPIC_OVERSAMPLE, seed=seed)
        chunks = _balanced_topic_chunks(pool, n_topic, seed=seed)
        for i, chunk in enumerate(chunks):
            cand = draft_topic_candidate(chunk, client, candidate_index=i)
            if cand is not None:
                candidates.append(cand)

    # Version-sensitive candidates
    if n_version_sensitive > 0:
        version_chunks = sampler.sample_matching(
            _VERSION_PATTERN, n_version_sensitive, seed=seed + 1
        )
        for i, chunk in enumerate(version_chunks):
            cand = draft_version_sensitive_candidate(chunk, client, candidate_index=i)
            if cand is not None:
                candidates.append(cand)

    # Unanswerable candidates
    for i in range(n_unanswerable):
        cand = draft_unanswerable_candidate(client, probe, candidate_index=i)
        if cand is not None:
            candidates.append(cand)

    return _dedupe_candidates(candidates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft golden-set expansion candidates.")
    parser.add_argument("--n-topic", type=int, default=80)
    parser.add_argument("--n-unanswerable", type=int, default=40)
    parser.add_argument("--n-version-sensitive", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queue-file", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = _OpenAIChatClient(model=args.model)
    sampler = _ChromaChunkSampler()
    probe = _ChromaRetrievalProbe()

    candidates = generate_candidates(
        client=client,
        sampler=sampler,
        probe=probe,
        n_topic=args.n_topic,
        n_unanswerable=args.n_unanswerable,
        n_version_sensitive=args.n_version_sensitive,
        seed=args.seed,
    )

    write_queue(candidates, args.queue_file)
    print(f"wrote {len(candidates)} candidates → {args.queue_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
