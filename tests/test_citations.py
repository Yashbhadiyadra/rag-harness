"""Tests for inline chunk-level citation parsing."""

from rag_harness.generation.citations import cited_chunk_indices, resolve_citations
from rag_harness.models import Chunk


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c::{i}",
        text=f"text {i}",
        source_file=f"doc-{i}.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=i,
    )


def test_cited_indices_extracts_unique_sorted() -> None:
    answer = "A Pod is a unit [1]. It wraps containers [2]. See also [1]."
    assert cited_chunk_indices(answer) == [1, 2]


def test_cited_indices_multi_digit() -> None:
    assert cited_chunk_indices("deep in the list [12] and [3]") == [3, 12]


def test_cited_indices_ignores_bare_numbers() -> None:
    # "port 8080" and "version 1.29" are not citation markers
    assert cited_chunk_indices("Use port 8080 on version 1.29") == []


def test_cited_indices_empty_when_none() -> None:
    assert cited_chunk_indices("An answer with no citations at all.") == []


def test_resolve_citations_maps_indices_to_chunks() -> None:
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    resolved = resolve_citations("First [1] then third [3].", chunks)
    assert [c.source_file for c in resolved] == ["doc-0.md", "doc-2.md"]


def test_resolve_citations_skips_out_of_range() -> None:
    chunks = [_chunk(0)]
    # [5] has no matching chunk - skipped, not an error
    resolved = resolve_citations("Cite [1] and [5].", chunks)
    assert len(resolved) == 1
    assert resolved[0].source_file == "doc-0.md"
