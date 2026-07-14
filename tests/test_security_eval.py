"""Tests for the security evaluation's pure parts (Phase 2 poison probe).

No LLM calls: context construction, canary detection, resistance stats,
and rendering are deterministic. The runner's LLM path is covered by
patching generate_async.
"""

from unittest.mock import patch

import pytest

from rag_harness.evaluation.security_eval import (
    CANARY,
    INJECTIONS,
    PoisonResult,
    answer_is_compromised,
    make_poisoned_context,
    render_markdown,
    run_poison_probe,
)
from rag_harness.models import GoldenCase

_CASE = GoldenCase(
    id="pods-001",
    question="What is a Pod?",
    reference_answer="A Pod is the smallest deployable unit in Kubernetes.",
    relevant_doc_ids=["content/en/docs/concepts/workloads/pods.md"],
)


def test_every_injection_carries_the_canary() -> None:
    for inj in INJECTIONS:
        assert CANARY in inj.text, f"{inj.name} lacks the canary payload"


def test_make_poisoned_context_splices_adversarial_in_middle() -> None:
    ctx = make_poisoned_context(_CASE, INJECTIONS[0])
    assert len(ctx) == 3
    # adversarial chunk is not first or last
    assert ctx[1].source_file == "attacker-controlled"
    assert ctx[0].source_file != "attacker-controlled"
    assert ctx[2].source_file != "attacker-controlled"
    # benign chunks carry the reference content
    assert _CASE.reference_answer in ctx[0].text


def test_make_poisoned_context_handles_no_doc_ids() -> None:
    case = GoldenCase(id="x", question="q", reference_answer="ans", relevant_doc_ids=[])
    ctx = make_poisoned_context(case, INJECTIONS[0])
    assert ctx[0].source_file == "synthetic"


def test_answer_is_compromised_detects_canary() -> None:
    assert answer_is_compromised(f"Sure, here you go: {CANARY}")
    assert answer_is_compromised(CANARY)
    assert not answer_is_compromised("A Pod is the smallest deployable unit.")


def test_resistance_rate() -> None:
    assert PoisonResult("x", n_cases=10, n_compromised=0).resistance_rate == 1.0
    assert PoisonResult("x", n_cases=10, n_compromised=3).resistance_rate == pytest.approx(0.7)
    assert PoisonResult("x", n_cases=0, n_compromised=0).resistance_rate == 1.0


def test_render_markdown_lists_every_injection() -> None:
    results = [
        PoisonResult("direct_override", 30, 0),
        PoisonResult("forged_system", 30, 2),
    ]
    md = render_markdown(results, commit="abc1234", timestamp="20260714T000000+0000")
    assert "direct_override" in md
    assert "forged_system" in md
    assert "abc1234" in md
    assert "100%" in md  # direct_override fully resisted


@pytest.mark.asyncio
async def test_run_poison_probe_counts_compromised() -> None:
    cases = [_CASE, _CASE]

    async def fake_generate(question: str, chunks: list) -> str:
        # simulate a vulnerable generator only for the forged_system template
        text = " ".join(c.text for c in chunks)
        if "SYSTEM OVERRIDE" in text:
            return f"Here is the answer {CANARY}"
        return "A Pod is the smallest deployable unit."

    with patch("rag_harness.evaluation.security_eval.generate_async", side_effect=fake_generate):
        results = await run_poison_probe(cases)

    by_name = {r.injection: r for r in results}
    assert by_name["forged_system"].n_compromised == 2  # both cases fell
    assert by_name["forged_system"].resistance_rate == 0.0
    assert by_name["direct_override"].n_compromised == 0
    assert by_name["direct_override"].resistance_rate == 1.0
