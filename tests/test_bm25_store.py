"""Unit tests for the BM25 tokenizer.

The BM25Store itself requires a populated ChromaDB collection and is exercised
in tests/integration/. Here we only test the pure tokenisation logic.
"""

from rag_harness.retrieval.bm25_store import tokenize


def test_tokenize_lowercases() -> None:
    assert tokenize("KUBECTL Apply") == ["kubectl", "apply"]


def test_tokenize_splits_on_punctuation() -> None:
    assert tokenize("pod-lifecycle.md") == ["pod", "lifecycle", "md"]


def test_tokenize_preserves_underscores() -> None:
    # \w includes underscore — 'foo_bar' stays intact
    assert tokenize("foo_bar baz") == ["foo_bar", "baz"]


def test_tokenize_handles_numbers() -> None:
    assert tokenize("kubernetes v1.32 release") == ["kubernetes", "v1", "32", "release"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_whitespace_only() -> None:
    assert tokenize("   \n\t  ") == []


def test_tokenize_kubernetes_identifiers() -> None:
    # Realistic K8s vocabulary — these must survive tokenisation
    text = "PodDisruptionBudget protects the ClusterRoleBinding"
    tokens = tokenize(text)
    assert "poddisruptionbudget" in tokens
    assert "clusterrolebinding" in tokens
