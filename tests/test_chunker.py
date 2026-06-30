import textwrap
from pathlib import Path

import pytest

from rag_harness.ingest.chunker import chunk_file, _update_heading_path


@pytest.fixture()
def tmp_md(tmp_path: Path) -> Path:
    """Write a small markdown file and return its path."""
    content = textwrap.dedent("""\
        ---
        title: Test Doc
        ---

        # Security

        Intro paragraph about security.

        ## RBAC

        Role-Based Access Control lets you configure permissions.

        ### Role Binding

        A RoleBinding grants permissions defined in a Role to a user.

        ## Network Policies

        Network policies restrict traffic between pods.
    """)
    f = tmp_path / "test.md"
    f.write_text(content)
    return f


def test_chunk_count(tmp_md: Path) -> None:
    chunks = chunk_file(tmp_md, tmp_md.parent, git_commit="abc123", doc_version="v1.29")
    # 4 headings with body text → 4 chunks
    assert len(chunks) == 4


def test_provenance_fields(tmp_md: Path) -> None:
    chunks = chunk_file(tmp_md, tmp_md.parent, git_commit="abc123", doc_version="v1.29")
    for chunk in chunks:
        assert chunk.git_commit == "abc123"
        assert chunk.doc_version == "v1.29"
        assert chunk.source_file.endswith("test.md")


def test_heading_path_hierarchy(tmp_md: Path) -> None:
    chunks = chunk_file(tmp_md, tmp_md.parent, git_commit="abc123", doc_version="v1.29")
    # chunk 0: # Security
    assert chunks[0].heading_path == ["Security"]
    # chunk 1: ## RBAC (under Security)
    assert chunks[1].heading_path == ["Security", "RBAC"]
    # chunk 2: ### Role Binding (under Security > RBAC)
    assert chunks[2].heading_path == ["Security", "RBAC", "Role Binding"]
    # chunk 3: ## Network Policies (sibling of RBAC — path resets to level 2)
    assert chunks[3].heading_path == ["Security", "Network Policies"]


def test_chunk_ids_are_unique(tmp_md: Path) -> None:
    chunks = chunk_file(tmp_md, tmp_md.parent, git_commit="abc123", doc_version="v1.29")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_frontmatter_stripped(tmp_md: Path) -> None:
    chunks = chunk_file(tmp_md, tmp_md.parent, git_commit="abc123", doc_version="v1.29")
    for chunk in chunks:
        assert "title:" not in chunk.text


def test_update_heading_path() -> None:
    path: list[str] = []
    path = _update_heading_path(path, 1, "Security")
    assert path == ["Security"]
    path = _update_heading_path(path, 2, "RBAC")
    assert path == ["Security", "RBAC"]
    path = _update_heading_path(path, 3, "Role Binding")
    assert path == ["Security", "RBAC", "Role Binding"]
    # Sibling at level 2 should drop "Role Binding"
    path = _update_heading_path(path, 2, "Network Policies")
    assert path == ["Security", "Network Policies"]
