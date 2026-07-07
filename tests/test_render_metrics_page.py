"""Tests for the static metrics-page generator (ADR-0010 §Public metrics page)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.render_metrics_page import main, render_page

from rag_harness.evaluation.history import HistoryEntry

# Deterministic timestamp used across tests
_FIXED_NOW = datetime(2026, 7, 5, 12, 0, 0, tzinfo=UTC)


def _entry(
    strategy: str,
    corrective: bool,
    correctness: float,
    cost: float,
    ts_offset_days: int = 0,
    *,
    p50_ms: float = 2000.0,
    p95_ms: float = 4500.0,
    per_case_scores: dict[str, list[float]] | None = None,
) -> HistoryEntry:
    """Build a plausible HistoryEntry for a synthetic history.

    ``per_case_scores`` is the ADR-0011 field. Default None simulates the
    pre-bootstrap era so existing tests keep exercising that path.
    """
    return HistoryEntry(
        timestamp=f"2026-07-{4 - ts_offset_days:02d}T09:00:00+00:00",
        git_commit=f"abc{ts_offset_days:03d}",
        strategy=strategy,
        corrective=corrective,
        n_cases=30,
        passed=correctness >= 0.7,
        mean_context_recall=0.75,
        mean_context_precision=0.85,
        mean_faithfulness=0.9,
        mean_correctness=correctness,
        mean_answer_relevancy=0.82,
        latency_p50_ms=p50_ms,
        latency_p95_ms=p95_ms,
        total_cost_usd=cost,
        per_case_scores=per_case_scores,
    )


def _make_scores(mean: float, n: int = 30, spread: float = 0.05) -> list[float]:
    """Deterministic per-case scores clustered around ``mean`` with a
    controlled spread. Used to synthesize per-case data in tests without
    involving the actual eval pipeline."""
    return [mean + spread * (i / (n - 1) - 0.5) * 2 for i in range(n)]


@pytest.fixture
def synthetic_history() -> list[HistoryEntry]:
    """6 entries: 3 combos × 2 timepoints. Enough for sparklines + corrective delta.

    Newest last (chronological order matches history file order).
    """
    return [
        # dense, corrective off — two runs, small improvement over time
        _entry("dense", False, 0.72, 0.0006, ts_offset_days=3),
        _entry("dense", False, 0.75, 0.0007, ts_offset_days=0),
        # dense, corrective on — better correctness, higher cost/latency
        _entry("dense", True, 0.80, 0.0015, ts_offset_days=3, p50_ms=3200.0),
        _entry("dense", True, 0.83, 0.0016, ts_offset_days=0, p50_ms=3300.0),
        # hybrid-rerank, corrective off — best latest correctness
        _entry("hybrid-rerank", False, 0.86, 0.0008, ts_offset_days=3),
        _entry("hybrid-rerank", False, 0.88, 0.0009, ts_offset_days=0),
    ]


# --- Empty history ----------------------------------------------------


def test_render_page_handles_empty_history() -> None:
    html_str = render_page([], _FIXED_NOW)
    assert "No runs yet" in html_str
    # No table or scatter rendered — assert those sections don't leak
    assert "Ablation table" not in html_str
    assert "Quality vs cost" not in html_str
    # Timestamp and run count still land in the footer
    assert "0 eval runs" in html_str
    assert "2026-07-05" in html_str


# --- Structural / self-contained checks -------------------------------


def test_render_page_is_self_contained(synthetic_history: list[HistoryEntry]) -> None:
    """No external asset references — no CDN, no remote fonts, no remote scripts.

    The metrics page must be renderable offline. This is what makes it
    honest — nothing in the report is deferred to a service we don't
    control.
    """
    html_str = render_page(synthetic_history, _FIXED_NOW)

    # Look for any src="http..." or href="http..." — external assets.
    external_ref = re.search(r'(src|href)="https?://', html_str)
    assert external_ref is None, f"external asset found: {external_ref.group()}"

    # Also assert no <script> tags at all (metrics page is JS-free).
    assert "<script" not in html_str


def test_render_page_declares_meta_and_style_blocks(
    synthetic_history: list[HistoryEntry],
) -> None:
    html_str = render_page(synthetic_history, _FIXED_NOW)
    assert "<!doctype html>" in html_str.lower()
    assert '<html lang="en">' in html_str
    assert "<title>RAG harness" in html_str
    assert 'name="robots"' in html_str  # noindex the page like the demo UI
    assert "<style>" in html_str


# --- Headline + config --------------------------------------------------


def test_headline_shows_latest_run_and_gate_status(
    synthetic_history: list[HistoryEntry],
) -> None:
    html_str = render_page(synthetic_history, _FIXED_NOW)
    # The latest entry in the fixture is hybrid-rerank, correctness=0.88, passed=True
    assert "gate passed" in html_str
    # Latest git commit tag from _entry() when ts_offset_days=0
    assert "abc000" in html_str


def test_headline_shows_production_config(
    synthetic_history: list[HistoryEntry],
) -> None:
    """Prod config (strategy=dense, corrective off) must appear even when the
    top-of-ablation row is a different combo."""
    html_str = render_page(synthetic_history, _FIXED_NOW)
    assert "Production config" in html_str
    # The exact prod line
    assert "dense · corrective off" in html_str


# --- Ablation table ---------------------------------------------------


def test_ablation_table_has_row_per_combo(
    synthetic_history: list[HistoryEntry],
) -> None:
    html_str = render_page(synthetic_history, _FIXED_NOW)
    # Table rows are tagged with strategy name in a <td>. Look for all three.
    assert ">dense<" in html_str or ">dense " in html_str or "dense<" in html_str
    assert "hybrid-rerank" in html_str
    # Corrective column values
    assert ">on<" in html_str
    assert ">off<" in html_str


def test_ablation_table_shows_latest_correctness_values(
    synthetic_history: list[HistoryEntry],
) -> None:
    """The number rendered in the correctness cell for each combo must match
    the latest entry for that combo (not an average, not the first entry)."""
    html_str = render_page(synthetic_history, _FIXED_NOW)
    # Latest values: dense/off=0.75, dense/on=0.83, hybrid-rerank/off=0.88
    assert "0.75" in html_str
    assert "0.83" in html_str
    assert "0.88" in html_str
    # First-run values must NOT be the ones displayed
    assert "0.72" not in html_str  # only appears in the sparkline path data
    # sparkline coords do not encode "0.72" as a substring in point pairs,
    # so a plain-string check is safe here.


def test_ablation_row_marked_production(synthetic_history: list[HistoryEntry]) -> None:
    """The dense + corrective-off row must carry the 'prod' badge."""
    html_str = render_page(synthetic_history, _FIXED_NOW)
    assert "prod-badge" in html_str


# --- Sparklines --------------------------------------------------------


def test_ablation_cells_include_inline_sparklines(
    synthetic_history: list[HistoryEntry],
) -> None:
    html_str = render_page(synthetic_history, _FIXED_NOW)
    # Each combo has ≥2 runs so sparklines should render as polylines.
    assert 'class="spark"' in html_str
    assert "<polyline" in html_str


# --- Scatter ----------------------------------------------------------


def test_scatter_section_present_and_labels_combos(
    synthetic_history: list[HistoryEntry],
) -> None:
    html_str = render_page(synthetic_history, _FIXED_NOW)
    assert "Quality vs cost" in html_str
    # Each combo produces one labeled dot.
    assert '<svg class="scatter"' in html_str
    assert "dense · corrective" in html_str  # from the corrective=True dot label
    assert "hybrid-rerank" in html_str


# --- Corrective delta panel -------------------------------------------


def test_corrective_delta_panel_reports_delta_for_dense(
    synthetic_history: list[HistoryEntry],
) -> None:
    """Latest dense/off correctness=0.75, dense/on=0.83 → +8.0pp.

    Positive delta must use the pass colour class, not the regress class.
    """
    html_str = render_page(synthetic_history, _FIXED_NOW)
    assert "Impact of corrective RAG" in html_str
    assert "+8.0pp" in html_str
    assert 'class="delta correct-delta">+8.0pp' in html_str


def test_corrective_delta_negative_uses_regress_class() -> None:
    """When corrective ON is worse than corrective OFF, the pp value must
    be marked with regress-delta (red), not correct-delta (green)."""
    entries = [
        _entry("dense", False, 0.85, 0.0006),
        _entry("dense", True, 0.75, 0.0016),  # corrective made things worse
    ]
    html_str = render_page(entries, _FIXED_NOW)
    assert 'class="delta regress-delta">-10.0pp' in html_str
    # Sanity: the "positive is good" class must NOT wrap this negative value
    assert 'class="delta correct-delta">-10.0pp' not in html_str


def test_corrective_delta_omits_sign_for_sub_precision_cost() -> None:
    """Deltas below display precision must not render as awkward '+< $0.001'."""
    entries = [
        _entry("dense", False, 0.7, 0.00050),
        _entry("dense", True, 0.72, 0.00051),  # d_cost = 0.00001 → "< $0.001"
    ]
    html_str = render_page(entries, _FIXED_NOW)
    # No awkward prefix concatenation
    assert "+< $0.001" not in html_str
    assert "−< $0.001" not in html_str
    # The magnitude still appears
    assert "< $0.001" in html_str


def test_corrective_delta_panel_skips_strategies_missing_one_side(
    synthetic_history: list[HistoryEntry],
) -> None:
    """hybrid-rerank has only corrective=off — it must not appear in the delta panel."""
    html_str = render_page(synthetic_history, _FIXED_NOW)
    # find the corrective-list block and assert hybrid-rerank isn't inside it
    match = re.search(r'<ul class="corrective-list">(.*?)</ul>', html_str, flags=re.DOTALL)
    assert match is not None, "corrective delta panel missing entirely"
    inside = match.group(1)
    assert "hybrid-rerank" not in inside


def test_corrective_delta_panel_omitted_when_no_dual_strategies() -> None:
    """When no strategy has both corrective states, the panel must not render."""
    entries = [
        _entry("dense", False, 0.7, 0.001),
        _entry("hybrid", False, 0.75, 0.001),
    ]
    html_str = render_page(entries, _FIXED_NOW)
    assert "Impact of corrective RAG" not in html_str


# --- ADR-0011: bootstrap CIs, pre-bootstrap labels, significance --------


def test_ablation_cell_shows_no_ci_label_pre_bootstrap(
    synthetic_history: list[HistoryEntry],
) -> None:
    """Rows without per_case_scores must show '(no CI)' with the
    no-ci class so mixed history reads as data-provenance, not a bug."""
    html_str = render_page(synthetic_history, _FIXED_NOW)
    assert 'class="no-ci"' in html_str
    assert "(no CI)" in html_str


def test_ablation_cell_shows_ci_brackets_when_per_case_scores_present() -> None:
    """A row with per_case_scores gets 95% bootstrap CI shown in
    square brackets alongside the point estimate."""
    entries = [
        _entry(
            "dense",
            False,
            0.75,
            0.0007,
            per_case_scores={
                "context_recall": _make_scores(0.77),
                "context_precision": _make_scores(0.85),
                "faithfulness": _make_scores(0.90),
                "correctness": _make_scores(0.75),
                "answer_relevancy": _make_scores(0.82),
            },
        )
    ]
    html_str = render_page(entries, _FIXED_NOW, seed=42)
    assert 'class="ci"' in html_str
    # Square-bracket interval with an en-dash between bounds
    assert re.search(r'class="ci">\[\d\.\d{2}–\d\.\d{2}\]</span>', html_str) is not None
    # A scored row must not have the no-ci CELL class rendered. The
    # intro paragraph legitimately mentions "(no CI)" verbatim to
    # explain what the label means, so we check for the class, not the
    # raw phrase.
    assert 'class="no-ci"' not in html_str


def test_corrective_delta_shows_significance_message_when_ci_overlaps_zero() -> None:
    """When the paired-difference CI overlaps zero, the panel must state
    it in plain language, not just colour the delta neutrally."""
    # on/off with essentially identical distributions → paired diffs
    # centre on zero, CI includes zero
    on_scores = _make_scores(0.80)
    off_scores = _make_scores(0.80)
    entries = [
        _entry(
            "dense",
            False,
            0.80,
            0.0007,
            per_case_scores={
                "context_recall": _make_scores(0.77),
                "context_precision": _make_scores(0.85),
                "faithfulness": _make_scores(0.90),
                "correctness": off_scores,
                "answer_relevancy": _make_scores(0.82),
            },
        ),
        _entry(
            "dense",
            True,
            0.80,
            0.0015,
            p50_ms=3200.0,
            per_case_scores={
                "context_recall": _make_scores(0.77),
                "context_precision": _make_scores(0.85),
                "faithfulness": _make_scores(0.90),
                "correctness": on_scores,
                "answer_relevancy": _make_scores(0.82),
            },
        ),
    ]
    html_str = render_page(entries, _FIXED_NOW, seed=42)
    assert "not statistically distinguishable from zero at n=30" in html_str
    # And the delta itself is rendered with the neutral indeterminate class,
    # not the green or red one
    assert "indeterminate-delta" in html_str


def test_corrective_delta_shows_ci_brackets_when_paired_diff_significant() -> None:
    """When the paired-difference CI excludes zero, the panel keeps the
    green/red colouring AND shows the CI bounds in brackets."""
    off_scores = _make_scores(0.70, spread=0.03)
    on_scores = [x + 0.20 for x in off_scores]  # uniform +20pp lift
    entries = [
        _entry(
            "dense",
            False,
            0.70,
            0.0007,
            per_case_scores={
                "context_recall": _make_scores(0.77),
                "context_precision": _make_scores(0.85),
                "faithfulness": _make_scores(0.90),
                "correctness": off_scores,
                "answer_relevancy": _make_scores(0.82),
            },
        ),
        _entry(
            "dense",
            True,
            0.90,
            0.0015,
            p50_ms=3200.0,
            per_case_scores={
                "context_recall": _make_scores(0.77),
                "context_precision": _make_scores(0.85),
                "faithfulness": _make_scores(0.90),
                "correctness": on_scores,
                "answer_relevancy": _make_scores(0.82),
            },
        ),
    ]
    html_str = render_page(entries, _FIXED_NOW, seed=42)
    # Should NOT say "not statistically distinguishable"
    assert "not statistically distinguishable" not in html_str
    # A CI bracket like [+X.Xpp, +Y.Ypp] is present next to the delta
    assert re.search(r'class="ci">\[[+\-−]\d+\.\dpp, [+\-−]\d+\.\dpp\]', html_str) is not None
    # Delta colour is green (positive effect)
    assert "correct-delta" in html_str


def test_corrective_delta_shows_no_ci_label_when_pre_bootstrap(
    synthetic_history: list[HistoryEntry],
) -> None:
    """Pre-bootstrap corrective rows must be labeled with the no-CI note
    in plain language, not silently coloured green/red only."""
    html_str = render_page(synthetic_history, _FIXED_NOW)
    # The em-dash is part of the label, but html.escape keeps the em-dash
    # verbatim so we look for the surrounding text.
    assert "no CI" in html_str
    assert "pre-bootstrap run" in html_str


# --- CLI entry point --------------------------------------------------


def test_main_writes_output_file(tmp_path: Path) -> None:
    """main() reads a specified history file and writes to a specified output path."""
    history = tmp_path / "history.jsonl"
    history.write_text(
        _entry("dense", False, 0.75, 0.0007).model_dump_json() + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out" / "metrics.html"

    rc = main(["--history", str(history), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "RAG harness — evaluation metrics" in body
    assert "1 eval run" in body  # singular
