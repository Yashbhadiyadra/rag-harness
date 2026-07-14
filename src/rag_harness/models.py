"""Shared data models: Chunk, GoldenCase, EvalResult, and EvalSummary."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single piece of text extracted from the K8s docs, with full provenance."""

    id: str  # unique: "{source_file}::{chunk_index}"
    text: str
    source_file: (
        str  # relative path in repo, e.g. "content/en/docs/concepts/security/rbac.md"  # noqa: E501
    )
    git_commit: str  # pinned commit hash this chunk was ingested from
    doc_version: str  # K8s release version, e.g. "v1.29"
    chunk_index: int  # position within the source file (0-based)
    heading_path: list[str] = Field(default_factory=list)  # heading hierarchy above this chunk


class GoldenCase(BaseModel):
    """A hand-verified evaluation case. Lives in evals/golden/ and is reviewed like code."""

    id: str
    question: str
    reference_answer: str
    relevant_doc_ids: list[str]  # source_file values that should appear in retrieved chunks


class EvalResult(BaseModel):
    """Scores for a single golden case after running the full RAG pipeline."""

    case_id: str
    question: str
    generated_answer: str
    retrieved_doc_ids: list[str]
    context_recall: float  # fraction of relevant_doc_ids that were retrieved
    context_precision: float = 0.0  # fraction of retrieved chunks that were useful
    faithfulness: float  # 0–1: is every claim in the answer grounded in context?
    correctness: float  # 0–1: does the answer match the reference answer?
    answer_relevancy: float = 0.0  # 0–1: is the answer on-topic (regardless of correctness)?
    # Operational metrics (added Phase 7)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Corrective RAG telemetry (added Phase 8; None on baseline runs)
    corrective_category: str | None = None
    corrective_attempts: int | None = None
    corrective_reformulated_query: str | None = None


class EvalSummary(BaseModel):
    """Aggregate scores across all golden cases. Gate fails if any metric is below threshold."""

    results: list[EvalResult]
    mean_context_recall: float
    mean_faithfulness: float
    mean_correctness: float
    passed: bool  # True only if all means are above their thresholds
    # Extended metrics (added Phase 7). Defaults preserve backward compat.
    mean_context_precision: float = 0.0
    mean_answer_relevancy: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Negative rejection (added Phase 4): of the golden cases that are
    # genuinely unanswerable (reference answer is the refusal), the fraction
    # the system correctly refused instead of improvising. Headline
    # reliability metric - answers when it should, abstains when it must.
    n_unanswerable: int = 0
    abstention_rate: float = 1.0
