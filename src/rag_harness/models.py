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
    faithfulness: float  # 0–1: is every claim in the answer grounded in context?
    correctness: float  # 0–1: does the answer match the reference answer?


class EvalSummary(BaseModel):
    """Aggregate scores across all golden cases. Gate fails if any metric is below threshold."""

    results: list[EvalResult]
    mean_context_recall: float
    mean_faithfulness: float
    mean_correctness: float
    passed: bool  # True only if all means are above their thresholds
