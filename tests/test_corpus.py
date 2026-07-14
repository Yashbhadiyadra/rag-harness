"""Tests for the corpus abstraction and corpus-agnostic loading."""

from pathlib import Path

import pytest

from rag_harness.ingest.corpus import CorpusSpec, default_corpus
from rag_harness.ingest.loader import load_doc_paths


def test_default_corpus_is_kubernetes() -> None:
    c = default_corpus()
    assert c.name == "k8s"
    assert "kubernetes/website" in c.repo_url
    assert c.docs_subpath == "content/en/docs"
    assert c.doc_glob == "*.md"


def test_local_dir_name_is_per_corpus() -> None:
    assert CorpusSpec("k8s", "u", "r", "s").local_dir_name == "k8s_docs"
    assert CorpusSpec("mydocs", "u", "r", "s").local_dir_name == "mydocs_docs"


def test_corpus_spec_is_frozen() -> None:
    c = CorpusSpec("k8s", "u", "r", "s")
    with pytest.raises(Exception):
        c.name = "other"  # type: ignore[misc]


def test_load_doc_paths_respects_a_custom_corpus(tmp_path: Path) -> None:
    # a non-K8s corpus layout: docs under "guides/", .markdown files
    docs = tmp_path / "guides"
    (docs / "sub").mkdir(parents=True)
    (docs / "a.markdown").write_text("# A")
    (docs / "sub" / "b.markdown").write_text("# B")
    (docs / "ignore.txt").write_text("not a doc")

    corpus = CorpusSpec(
        name="mydocs", repo_url="u", git_ref="r", docs_subpath="guides", doc_glob="*.markdown"
    )
    paths = load_doc_paths(tmp_path, corpus)
    names = sorted(p.name for p in paths)
    assert names == ["a.markdown", "b.markdown"]  # .txt excluded by the glob


def test_load_doc_paths_missing_dir_raises(tmp_path: Path) -> None:
    corpus = CorpusSpec("x", "u", "r", "nonexistent")
    with pytest.raises(FileNotFoundError):
        load_doc_paths(tmp_path, corpus)
