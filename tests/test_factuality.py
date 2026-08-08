"""Tests for the factuality gateway and its corrective-loop wiring (ADR-0029)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.config import settings
from rag_harness.evaluation.claim_eval import ClaimLabel
from rag_harness.generation.corrective import corrective_generate_async
from rag_harness.generation.critic import Category
from rag_harness.generation.factuality import GatewayResult, factuality_gateway
from rag_harness.models import Chunk


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        text=f"text for {cid}",
        source_file=f"docs/{cid}.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
        heading_path=[],
    )


@pytest.mark.asyncio
async def test_gateway_passes_through_when_all_claims_grounded() -> None:
    labels = [ClaimLabel("A Pod wraps containers.", "grounded")]
    with patch(
        "rag_harness.generation.factuality.classify_claims",
        new=AsyncMock(return_value=labels),
    ):
        result = await factuality_gateway("q?", "A Pod wraps containers.", [_chunk("a")])

    assert result == GatewayResult(
        answer="A Pod wraps containers.", revised=False, n_claims=1, n_flagged=0
    )


@pytest.mark.asyncio
async def test_gateway_regenerates_when_a_claim_is_unsupported() -> None:
    labels = [
        ClaimLabel("A Pod wraps containers.", "grounded"),
        ClaimLabel("Pods autoscale by default.", "contradicted"),
    ]
    with (
        patch(
            "rag_harness.generation.factuality.classify_claims",
            new=AsyncMock(return_value=labels),
        ),
        patch(
            "rag_harness.generation.factuality._regenerate",
            new=AsyncMock(return_value="A Pod wraps containers [1]."),
        ) as regen,
    ):
        result = await factuality_gateway("q?", "draft", [_chunk("a")])

    assert result.revised is True
    assert result.n_flagged == 1
    assert result.answer == "A Pod wraps containers [1]."
    # the flagged claim text is fed back to the regeneration step
    assert "Pods autoscale by default." in regen.call_args.args[2]


@pytest.mark.asyncio
async def test_corrective_loop_runs_gateway_only_when_enabled() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("a")])
    critic = MagicMock()
    critic.score_batch_async = AsyncMock(return_value=[0.9])
    critic.categorise.return_value = Category.CORRECT
    critic._incorrect_threshold = 0.3

    gateway_ret = GatewayResult(answer="revised answer", revised=True, n_claims=2, n_flagged=1)
    with (
        patch(
            "rag_harness.generation.corrective.generate_async",
            new=AsyncMock(return_value="draft answer"),
        ),
        patch(
            "rag_harness.generation.factuality.factuality_gateway",
            new=AsyncMock(return_value=gateway_ret),
        ),
        patch.object(settings, "factuality_gateway_enabled", True),
    ):
        result = await corrective_generate_async("q?", retriever, critic=critic)

    assert result.answer == "revised answer"
    assert result.factuality_revised is True


@pytest.mark.asyncio
async def test_corrective_loop_skips_gateway_when_disabled() -> None:
    retriever = MagicMock()
    retriever.retrieve_async = AsyncMock(return_value=[_chunk("a")])
    critic = MagicMock()
    critic.score_batch_async = AsyncMock(return_value=[0.9])
    critic.categorise.return_value = Category.CORRECT
    critic._incorrect_threshold = 0.3

    gateway = AsyncMock()
    with (
        patch(
            "rag_harness.generation.corrective.generate_async",
            new=AsyncMock(return_value="draft answer"),
        ),
        patch("rag_harness.generation.factuality.factuality_gateway", new=gateway),
        patch.object(settings, "factuality_gateway_enabled", False),
    ):
        result = await corrective_generate_async("q?", retriever, critic=critic)

    assert result.answer == "draft answer"
    assert result.factuality_revised is False
    gateway.assert_not_called()
