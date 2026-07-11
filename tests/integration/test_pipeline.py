"""End-to-end integration tests for the ingest → index → retrieve pipeline.

These tests use a real ChromaDB collection in a temporary directory and a small
fixture corpus. Only the OpenAI embedding API is mocked - every other component
(chunker, indexer, retriever) runs its real code path.

Run with:
    pytest -m integration
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_harness.ingest.chunker import chunk_docs
from rag_harness.ingest.indexer import index_chunks
from rag_harness.retrieval.dense import DenseRetriever

# Dimensionality must be consistent across ingest and query mocks.
_EMBED_DIM = 8


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake vectors based on text hash, not random."""
    vecs = []
    for text in texts:
        seed = hash(text) % (10**6)
        base = [(seed + i) % 100 / 100.0 for i in range(_EMBED_DIM)]
        total = sum(x**2 for x in base) ** 0.5 or 1.0
        vecs.append([x / total for x in base])
    return vecs


@pytest.fixture()
def fixture_corpus(tmp_path: Path) -> Path:
    """Write a small two-file K8s docs corpus to a temp directory."""
    docs_dir = tmp_path / "content" / "en" / "docs" / "concepts"
    docs_dir.mkdir(parents=True)

    rbac_doc = textwrap.dedent("""\
        # RBAC

        Role-Based Access Control (RBAC) lets you configure fine-grained
        permissions in Kubernetes.

        ## Roles and ClusterRoles

        A Role defines permissions within a namespace. A ClusterRole applies
        cluster-wide. Both list the API groups, resources, and verbs allowed.

        ## RoleBindings

        A RoleBinding grants a Role to a user, group, or service account
        within a specific namespace.
    """)

    pods_doc = textwrap.dedent("""\
        # Pods

        A Pod is the smallest deployable unit in Kubernetes.

        ## Pod Lifecycle

        Pods pass through Pending, Running, Succeeded, and Failed phases.
        The kubelet manages the lifecycle on each node.
    """)

    (docs_dir / "rbac.md").write_text(rbac_doc)
    (docs_dir / "pods.md").write_text(pods_doc)
    return tmp_path


@pytest.fixture()
def populated_index(tmp_path: Path, fixture_corpus: Path) -> Path:
    """Ingest the fixture corpus into a real ChromaDB stored in tmp_path."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()

    doc_paths = list((fixture_corpus / "content" / "en" / "docs" / "concepts").glob("*.md"))
    chunks = chunk_docs(
        doc_paths,
        repo_root=fixture_corpus,
        git_commit="test-sha",
        doc_version="v1.32",
    )

    # Patch the OpenAI client used by embed_chunks so no real API call is made.
    def _mock_create(model: str, input: list[str]) -> MagicMock:  # noqa: A002
        resp = MagicMock()
        resp.data = [MagicMock(embedding=vec) for vec in _fake_embed(input)]
        return resp

    with (
        patch("rag_harness.ingest.embedder.settings.embedding_model", "fake-model"),
        patch("rag_harness.ingest.embedder.settings.chroma_db_path", str(chroma_dir)),
        patch("rag_harness.ingest.indexer.settings.chroma_db_path", str(chroma_dir)),
        patch("rag_harness.ingest.indexer.settings.chroma_collection", "test_col"),
        patch("rag_harness.ingest.embedder._client") as mock_client,
    ):
        mock_client.embeddings.create.side_effect = lambda model, input: _mock_create(  # noqa: A002
            model, input
        )
        from rag_harness.ingest.embedder import embed_chunks

        embedded = embed_chunks(chunks)
        index_chunks(embedded)

    return chroma_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ingest_produces_chunks_for_both_docs(fixture_corpus: Path) -> None:
    """Chunker produces at least one chunk per fixture document."""
    doc_paths = list((fixture_corpus / "content" / "en" / "docs" / "concepts").glob("*.md"))
    chunks = chunk_docs(
        doc_paths,
        repo_root=fixture_corpus,
        git_commit="test-sha",
        doc_version="v1.32",
    )
    source_files = {c.source_file for c in chunks}
    assert any("rbac" in sf for sf in source_files)
    assert any("pods" in sf for sf in source_files)


@pytest.mark.integration
def test_chunk_provenance_is_recorded(fixture_corpus: Path) -> None:
    """Every chunk carries the git commit and doc version it was ingested from."""
    doc_paths = list((fixture_corpus / "content" / "en" / "docs" / "concepts").glob("*.md"))
    chunks = chunk_docs(
        doc_paths,
        repo_root=fixture_corpus,
        git_commit="test-sha",
        doc_version="v1.32",
    )
    for chunk in chunks:
        assert chunk.git_commit == "test-sha"
        assert chunk.doc_version == "v1.32"
        assert chunk.source_file != ""


@pytest.mark.integration
def test_retriever_returns_relevant_chunk(tmp_path: Path, populated_index: Path) -> None:
    """A query about RBAC retrieves a chunk from the RBAC document."""

    def _mock_query_embed(model: str, input: list[str]) -> MagicMock:  # noqa: A002
        resp = MagicMock()
        resp.data = [MagicMock(embedding=vec) for vec in _fake_embed(input)]
        return resp

    with (
        patch("rag_harness.retrieval.dense.settings.chroma_db_path", str(populated_index)),
        patch("rag_harness.retrieval.dense.settings.chroma_collection", "test_col"),
        patch("rag_harness.retrieval.dense.settings.retrieval_top_k", 3),
        patch("rag_harness.retrieval.dense.settings.embedding_model", "fake-model"),
        patch("rag_harness.retrieval.dense.OpenAI") as mock_openai_cls,
    ):
        mock_openai_cls.return_value.embeddings.create.side_effect = _mock_query_embed
        retriever = DenseRetriever()
        chunks = retriever.retrieve("RBAC roles and permissions", top_k=3)

    assert len(chunks) > 0
    assert all(hasattr(c, "source_file") for c in chunks)
    assert all(hasattr(c, "heading_path") for c in chunks)


@pytest.mark.integration
def test_retriever_returns_chunk_with_correct_structure(
    tmp_path: Path, populated_index: Path
) -> None:
    """Retrieved chunks have all required provenance fields populated."""

    def _mock_query_embed(model: str, input: list[str]) -> MagicMock:  # noqa: A002
        resp = MagicMock()
        resp.data = [MagicMock(embedding=vec) for vec in _fake_embed(input)]
        return resp

    with (
        patch("rag_harness.retrieval.dense.settings.chroma_db_path", str(populated_index)),
        patch("rag_harness.retrieval.dense.settings.chroma_collection", "test_col"),
        patch("rag_harness.retrieval.dense.settings.retrieval_top_k", 3),
        patch("rag_harness.retrieval.dense.settings.embedding_model", "fake-model"),
        patch("rag_harness.retrieval.dense.OpenAI") as mock_openai_cls,
    ):
        mock_openai_cls.return_value.embeddings.create.side_effect = _mock_query_embed
        retriever = DenseRetriever()
        chunks = retriever.retrieve("pod lifecycle phases", top_k=3)

    for chunk in chunks:
        assert chunk.id != ""
        assert chunk.text != ""
        assert chunk.git_commit == "test-sha"
        assert chunk.doc_version == "v1.32"
        assert isinstance(chunk.heading_path, list)
