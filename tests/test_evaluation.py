import asyncio
import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_harness.evaluation.metrics import (
    _parse_letter,
    answer_relevancy,
    context_precision,
    context_recall,
    correctness_letter_async,
)
from rag_harness.evaluation.runner import (
    _percentile,
    export_results,
    load_golden_cases,
    run_eval,
)
from rag_harness.models import Chunk, EvalResult, EvalSummary


def _make_chunk(source_file: str) -> Chunk:
    return Chunk(
        id=f"{source_file}::0",
        text="Some content.",
        source_file=source_file,
        git_commit="abc123",
        doc_version="v1.29",
        chunk_index=0,
        heading_path=[],
    )


# --- context_recall (deterministic, no mocks needed) ---


def test_context_recall_full_hit() -> None:
    chunks = [_make_chunk("content/en/docs/security/rbac.md")]
    score = context_recall(chunks, ["content/en/docs/security/rbac.md"])
    assert score == 1.0


def test_context_recall_partial_hit() -> None:
    chunks = [_make_chunk("content/en/docs/security/rbac.md")]
    score = context_recall(
        chunks,
        [
            "content/en/docs/security/rbac.md",
            "content/en/docs/concepts/overview.md",
        ],
    )
    assert score == 0.5


def test_context_recall_no_hit() -> None:
    chunks = [_make_chunk("content/en/docs/networking/services.md")]
    score = context_recall(chunks, ["content/en/docs/security/rbac.md"])
    assert score == 0.0


def test_context_recall_empty_relevant() -> None:
    # No relevant docs specified → perfect score by convention
    score = context_recall([], [])
    assert score == 1.0


# --- load_golden_cases ---


def test_load_golden_cases(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "test.json").write_text(json.dumps(data))
    cases = load_golden_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].id == "test-001"


# --- run_eval gate ---


def test_run_eval_passes_when_above_thresholds(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(
        return_value=[_make_chunk("content/en/docs/security/rbac.md")]
    )

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="Role-Based Access Control.",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.95,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.90,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.8,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path))

    assert summary.passed is True
    assert summary.mean_context_recall == 1.0
    # New metrics carried through into summary
    assert summary.mean_answer_relevancy == 0.9
    assert summary.mean_context_precision == 0.8


def test_run_eval_reports_negative_rejection(tmp_path: Path) -> None:
    # One answerable case and one genuinely unanswerable case (reference is the
    # refusal). The generator is stubbed to refuse on both, so the answerable
    # case scores poorly but the unanswerable one is a correct abstention.
    data = [
        {
            "id": "answerable-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        },
        {
            "id": "unanswerable-001",
            "question": "What is the airspeed of an unladen swallow?",
            "reference_answer": (
                "I do not have enough information in the provided context to answer this question."
            ),
            "relevant_doc_ids": [],
        },
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(
        return_value=[_make_chunk("content/en/docs/security/rbac.md")]
    )

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="I do not have enough information in the provided context.",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.5,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.5,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.5,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path))

    # Only the one unanswerable case counts; the generator refused it correctly.
    assert summary.n_unanswerable == 1
    assert summary.abstention_rate == 1.0


def test_run_eval_negative_rejection_defaults_when_all_answerable(tmp_path: Path) -> None:
    data = [
        {
            "id": "answerable-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(
        return_value=[_make_chunk("content/en/docs/security/rbac.md")]
    )

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="Role-Based Access Control.",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.95,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.90,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.8,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path))

    # No unanswerable cases in the set: rate defaults to 1.0, count is 0.
    assert summary.n_unanswerable == 0
    assert summary.abstention_rate == 1.0


def test_run_eval_fails_when_below_threshold(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["content/en/docs/security/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[])  # retrieval miss → recall=0.0

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="I do not know.",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.90,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.80,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.5,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.0,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path))

    assert summary.passed is False
    assert summary.mean_context_recall == 0.0


# --- export_results ---


def _make_summary() -> EvalSummary:
    result = EvalResult(
        case_id="test-001",
        question="What is RBAC?",
        generated_answer="Role-Based Access Control.",
        retrieved_doc_ids=["content/en/docs/security/rbac.md"],
        context_recall=1.0,
        faithfulness=0.95,
        correctness=0.90,
    )
    return EvalSummary(
        results=[result],
        mean_context_recall=1.0,
        mean_faithfulness=0.95,
        mean_correctness=0.90,
        passed=True,
    )


def test_export_results_json(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    export_results(_make_summary(), out)
    data = json.loads(out.read_text())
    assert data["passed"] is True
    assert data["mean_context_recall"] == 1.0
    # New operational fields are included in the JSON export
    assert "latency_p50_ms" in data
    assert "latency_p95_ms" in data
    assert "total_cost_usd" in data
    assert "mean_answer_relevancy" in data
    assert "mean_context_precision" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["case_id"] == "test-001"


def test_export_results_csv(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    export_results(_make_summary(), out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "test-001"
    assert float(rows[0]["context_recall"]) == 1.0
    assert float(rows[0]["faithfulness"]) == 0.95


def test_export_results_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        export_results(_make_summary(), tmp_path / "results.txt")


# --- answer_relevancy ---


def _mock_llm_score(score: str) -> MagicMock:
    """Return a mock AsyncOpenAI client whose async .create returns a raw score string."""
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = score
    resp.usage = MagicMock(prompt_tokens=50, completion_tokens=3)
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def test_parse_letter_maps_grades() -> None:
    assert _parse_letter("A") == 1.0
    assert _parse_letter("B") == 0.75
    assert _parse_letter("C") == 0.5
    assert _parse_letter("D") == 0.25
    assert _parse_letter("E") == 0.0
    assert _parse_letter(" b ") == 0.75  # tolerant of whitespace/case
    assert _parse_letter("Grade: A") == 0.0  # not a bare letter -> default


@pytest.mark.asyncio
async def test_correctness_letter_scores_via_letter_prompt() -> None:
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("B")):
        score = await correctness_letter_async(
            "What is a Pod?", "A pod runs containers.", "A pod is a group of containers."
        )
    assert score == 0.75


def test_answer_relevancy_parses_score() -> None:
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("0.85")):
        score = answer_relevancy("What is RBAC?", "RBAC controls permissions.")
    assert score == 0.85


def test_answer_relevancy_empty_answer_returns_zero() -> None:
    # No LLM call needed for empty answer - short-circuit
    with patch("rag_harness.evaluation.metrics._client") as mock_client:
        score = answer_relevancy("What is RBAC?", "")
    assert score == 0.0
    mock_client.chat.completions.create.assert_not_called()


def test_answer_relevancy_whitespace_only_returns_zero() -> None:
    with patch("rag_harness.evaluation.metrics._client") as mock_client:
        score = answer_relevancy("What is RBAC?", "   \n   ")
    assert score == 0.0
    mock_client.chat.completions.create.assert_not_called()


def test_answer_relevancy_clamps_out_of_range() -> None:
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("1.7")):
        assert answer_relevancy("Q?", "A.") == 1.0
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("-0.3")):
        assert answer_relevancy("Q?", "A.") == 0.0


# --- context_precision ---


def test_context_precision_parses_score() -> None:
    chunks = [_make_chunk("content/en/docs/security/rbac.md")]
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("0.6")):
        score = context_precision("What is RBAC?", chunks, "RBAC controls permissions.")
    assert score == 0.6


def test_context_precision_empty_chunks_returns_zero() -> None:
    with patch("rag_harness.evaluation.metrics._client") as mock_client:
        score = context_precision("Q?", [], "reference")
    assert score == 0.0
    # No LLM call when there are no chunks to score
    mock_client.chat.completions.create.assert_not_called()


def test_context_precision_includes_all_chunks_in_prompt() -> None:
    chunks = [
        _make_chunk("docs/a.md"),
        _make_chunk("docs/b.md"),
        _make_chunk("docs/c.md"),
    ]
    with patch("rag_harness.evaluation.metrics._client", _mock_llm_score("0.5")) as mc:
        context_precision("Q?", chunks, "ref")
    # Verify all chunk texts appear in the LLM prompt (order doesn't matter for this test)
    call_kwargs = mc.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][1]["content"]
    assert "[1]" in user_msg and "[2]" in user_msg and "[3]" in user_msg


def test_context_precision_handles_non_numeric_response() -> None:
    chunks = [_make_chunk("docs/a.md")]
    with patch(
        "rag_harness.evaluation.metrics._client",
        _mock_llm_score("this is not a number"),
    ):
        score = context_precision("Q?", chunks, "ref")
    # Falls back to 0.0 per the LLM judge parsing contract
    assert score == 0.0


# --- _percentile ---


def test_percentile_empty_list_returns_zero() -> None:
    assert _percentile([], 50) == 0.0


def test_percentile_single_element() -> None:
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 95) == 42.0


def test_percentile_odd_length_p50() -> None:
    # 5 elements: nearest-rank rank(50) = ceil(0.5 * 5) = 3 → index 2 → 30
    assert _percentile([10, 20, 30, 40, 50], 50) == 30


def test_percentile_p95_on_20_samples() -> None:
    # 20 elements 1..20: rank(95) = ceil(0.95 * 20) = 19 → index 18 → 19
    assert _percentile([float(i) for i in range(1, 21)], 95) == 19


def test_percentile_unsorted_input() -> None:
    # Percentile is computed on sorted values regardless of input order
    assert _percentile([50, 10, 30, 40, 20], 50) == 30


# --- run_eval operational aggregation ---


# --- corrective mode in run_eval ---


def test_run_eval_corrective_routes_through_corrective_generate(tmp_path: Path) -> None:
    from rag_harness.generation.corrective import CorrectiveResult
    from rag_harness.generation.critic import Category

    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["docs/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    fake_chunk = _make_chunk("docs/rbac.md")
    fake_result = CorrectiveResult(
        answer="Role-Based Access Control.",
        chunks_used=[fake_chunk],
        category=Category.CORRECT,
        attempts=1,
        scores=[0.9],
        reformulated_query=None,
    )

    with (
        patch(
            "rag_harness.evaluation.runner.corrective_generate_async",
            new_callable=AsyncMock,
            return_value=fake_result,
        ) as mock_corrective,
        patch(
            "rag_harness.evaluation.runner.generate_async", new_callable=AsyncMock
        ) as mock_generate,
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path, use_corrective=True))

    # Corrective path taken; baseline generate not called
    mock_corrective.assert_called_once()
    mock_generate.assert_not_called()

    # Corrective telemetry populated on the EvalResult
    result = summary.results[0]
    assert result.corrective_category == "correct"
    assert result.corrective_attempts == 1
    assert result.corrective_reformulated_query is None


def test_run_eval_corrective_off_uses_baseline_generate(tmp_path: Path) -> None:
    data = [
        {
            "id": "test-001",
            "question": "What is RBAC?",
            "reference_answer": "Role-Based Access Control.",
            "relevant_doc_ids": ["docs/rbac.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk("docs/rbac.md")])

    with (
        patch(
            "rag_harness.evaluation.runner.corrective_generate_async", new_callable=AsyncMock
        ) as mock_corrective,
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="answer",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path, use_corrective=False))

    mock_corrective.assert_not_called()
    # Baseline telemetry fields stay None
    result = summary.results[0]
    assert result.corrective_category is None
    assert result.corrective_attempts is None
    assert result.corrective_reformulated_query is None


def test_run_eval_corrective_captures_reformulation(tmp_path: Path) -> None:
    from rag_harness.generation.corrective import CorrectiveResult
    from rag_harness.generation.critic import Category

    data = [
        {
            "id": "test-001",
            "question": "Q?",
            "reference_answer": "A.",
            "relevant_doc_ids": ["docs/a.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    fake_result = CorrectiveResult(
        answer="answer after retry",
        chunks_used=[_make_chunk("docs/a.md")],
        category=Category.CORRECT,
        attempts=2,
        scores=[0.8],
        reformulated_query="better keywords Q?",
    )

    with (
        patch(
            "rag_harness.evaluation.runner.corrective_generate_async",
            new_callable=AsyncMock,
            return_value=fake_result,
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
    ):
        summary = asyncio.run(run_eval(MagicMock(), golden_dir=tmp_path, use_corrective=True))

    r = summary.results[0]
    assert r.corrective_attempts == 2
    assert r.corrective_reformulated_query == "better keywords Q?"


def test_run_eval_corrective_refusal_still_scored(tmp_path: Path) -> None:
    from rag_harness.generation.corrective import NO_INFO_MESSAGE, CorrectiveResult
    from rag_harness.generation.critic import Category

    data = [
        {
            "id": "test-001",
            "question": "Q?",
            "reference_answer": "A.",
            "relevant_doc_ids": ["docs/a.md"],
        }
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    # Refusal path: no chunks used, standard NO_INFO_MESSAGE
    fake_result = CorrectiveResult(
        answer=NO_INFO_MESSAGE,
        chunks_used=[],
        category=Category.INCORRECT,
        attempts=2,
        scores=[0.1],
        reformulated_query="rewritten Q?",
    )

    with (
        patch(
            "rag_harness.evaluation.runner.corrective_generate_async",
            new_callable=AsyncMock,
            return_value=fake_result,
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.5,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.0,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.0,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.0,
        ),
    ):
        summary = asyncio.run(run_eval(MagicMock(), golden_dir=tmp_path, use_corrective=True))

    r = summary.results[0]
    assert r.generated_answer == NO_INFO_MESSAGE
    assert r.retrieved_doc_ids == []
    assert r.corrective_category == "incorrect"
    # A refusal correctly scored low on correctness - that's the intended signal
    assert r.correctness == 0.0


def test_run_eval_case_filter_restricts_to_ids(tmp_path: Path) -> None:
    def _case(cid: str, n: int) -> dict:
        return {
            "id": cid,
            "question": f"Q{n}?",
            "reference_answer": f"A{n}.",
            "relevant_doc_ids": ["a"],
        }

    data = [_case("keep", 1), _case("drop", 2), _case("also-keep", 3)]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk("a")])

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="ans",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
    ):
        summary = asyncio.run(
            run_eval(
                mock_retriever,
                golden_dir=tmp_path,
                case_filter=["keep", "also-keep"],
            )
        )

    ids = {r.case_id for r in summary.results}
    assert ids == {"keep", "also-keep"}


def test_run_eval_case_filter_none_runs_all_cases(tmp_path: Path) -> None:
    data = [
        {"id": f"c{i}", "question": "Q?", "reference_answer": "A.", "relevant_doc_ids": ["a"]}
        for i in range(3)
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk("a")])

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            return_value="ans",
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path, case_filter=None))

    assert len(summary.results) == 3


def test_run_eval_case_filter_no_matches_raises(tmp_path: Path) -> None:
    data = [
        {"id": "real", "question": "Q?", "reference_answer": "A.", "relevant_doc_ids": ["a"]},
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    with pytest.raises(ValueError, match="No golden cases"):
        asyncio.run(run_eval(MagicMock(), golden_dir=tmp_path, case_filter=["nonexistent"]))


def test_run_eval_aggregates_operational_metrics(tmp_path: Path) -> None:
    data = [
        {
            "id": f"test-{i}",
            "question": f"Q{i}?",
            "reference_answer": f"A{i}.",
            "relevant_doc_ids": ["docs/a.md"],
        }
        for i in range(3)
    ]
    (tmp_path / "cases.json").write_text(json.dumps(data))

    mock_retriever = MagicMock()
    mock_retriever.retrieve_async = AsyncMock(return_value=[_make_chunk("docs/a.md")])

    # Simulate a real call recording usage - the evaluate_case wrapper opens
    # its own collect_usage() block, so we need something inside to record.
    # Easiest: patch the metric functions to return numbers but also stub the
    # generation LLM path to inject a usage record via the ContextVar.

    from rag_harness.observability.usage import TokenUsage, record_usage

    def _generate_stub(question: str, chunks: list) -> str:
        # Each evaluate_case call records one 100-token record via generate
        record_usage(TokenUsage("gpt-4o-mini", 100, 20, 0.030))
        return f"answer for {question}"

    with (
        patch(
            "rag_harness.evaluation.runner.generate_async",
            new_callable=AsyncMock,
            side_effect=_generate_stub,
        ),
        patch(
            "rag_harness.evaluation.runner.faithfulness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.correctness_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.answer_relevancy_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
        patch(
            "rag_harness.evaluation.runner.context_precision_async",
            new_callable=AsyncMock,
            return_value=0.9,
        ),
    ):
        summary = asyncio.run(run_eval(mock_retriever, golden_dir=tmp_path))

    # 3 cases × (100 in, 20 out, $0.030)
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 60
    assert round(summary.total_cost_usd, 6) == 0.090
    # Latency was measured (non-zero) but exact value depends on wall clock
    assert summary.latency_p50_ms > 0.0
    assert summary.latency_p95_ms >= summary.latency_p50_ms
