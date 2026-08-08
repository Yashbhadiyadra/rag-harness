"""Tests for the claim-level groundedness probe's pure parts and wiring (ADR-0027)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.evaluation.claim_eval import (
    ClaimEvalResult,
    ClaimLabel,
    _parse_claims,
    aggregate,
    run_claim_eval,
)
from rag_harness.models import Chunk, GoldenCase


def test_parse_claims_valid_json() -> None:
    raw = (
        '{"claims": [{"claim": "A", "label": "grounded"}, {"claim": "B", "label": "contradicted"}]}'
    )
    labels = _parse_claims(raw)
    assert labels == [ClaimLabel("A", "grounded"), ClaimLabel("B", "contradicted")]


def test_parse_claims_strips_code_fence() -> None:
    raw = '```json\n{"claims": [{"claim": "A", "label": "GROUNDED"}]}\n```'
    labels = _parse_claims(raw)
    assert labels == [ClaimLabel("A", "grounded")]  # label lowercased


def test_parse_claims_unknown_label_coerced_to_ungrounded() -> None:
    raw = '{"claims": [{"claim": "A", "label": "maybe"}]}'
    assert _parse_claims(raw) == [ClaimLabel("A", "ungrounded")]


def test_parse_claims_malformed_returns_empty() -> None:
    assert _parse_claims("not json at all") == []
    assert _parse_claims('{"wrong_key": []}') == []


def test_parse_claims_skips_entries_missing_fields() -> None:
    raw = '{"claims": [{"claim": "A", "label": "grounded"}, {"label": "grounded"}]}'
    assert _parse_claims(raw) == [ClaimLabel("A", "grounded")]


def test_result_scores() -> None:
    r = ClaimEvalResult(
        n_cases=2, n_grounded=6, n_ungrounded=2, n_contradicted=1, n_complementary=1
    )
    assert r.n_claims == 10
    assert r.groundedness == pytest.approx(0.6)
    assert r.hallucination_rate == pytest.approx(0.3)  # (2 + 1) / 10


def test_result_zero_guard() -> None:
    r = ClaimEvalResult(
        n_cases=0, n_grounded=0, n_ungrounded=0, n_contradicted=0, n_complementary=0
    )
    assert r.groundedness == 0.0
    assert r.hallucination_rate == 0.0


def test_aggregate_counts_across_cases() -> None:
    per_case = [
        [ClaimLabel("a", "grounded"), ClaimLabel("b", "ungrounded")],
        [ClaimLabel("c", "grounded"), ClaimLabel("d", "complementary")],
    ]
    r = aggregate(per_case)
    assert r.n_cases == 2
    assert (r.n_grounded, r.n_ungrounded, r.n_contradicted, r.n_complementary) == (2, 1, 0, 1)


def _chunk() -> Chunk:
    return Chunk(
        id="c::0",
        text="A Pod wraps one or more containers.",
        source_file="pods.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
    )


@pytest.mark.asyncio
async def test_run_claim_eval_classifies_generated_answers() -> None:
    case = GoldenCase(
        id="c1", question="What is a Pod?", reference_answer="ref", relevant_doc_ids=[]
    )
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk()])

    reply = '{"claims": [{"claim": "A Pod wraps containers.", "label": "grounded"}]}'
    with (
        patch(
            "rag_harness.generation.generator.generate_async",
            new=AsyncMock(return_value="A Pod wraps containers."),
        ),
        patch(
            "rag_harness.evaluation.claim_eval._llm_raw_async",
            new=AsyncMock(return_value=reply),
        ),
    ):
        result = await run_claim_eval([case], retriever)

    assert result.n_cases == 1
    assert result.n_grounded == 1
    assert result.groundedness == 1.0
