import logging
import re
from pathlib import Path

import tiktoken

from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
_MAX_TOKENS = 512

# cl100k_base is the encoding used by text-embedding-3-small
_encoder = tiktoken.get_encoding("cl100k_base")


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip()


def _token_count(text: str) -> int:
    return len(_encoder.encode(text))


def _update_heading_path(current: list[str], level: int, heading: str) -> list[str]:
    # Truncate path to parent level then append — keeps hierarchy consistent.
    # e.g. current=["Security", "RBAC"], level=2, heading="Roles" → ["Security", "Roles"]
    return current[: level - 1] + [heading]


def chunk_file(
    path: Path,
    repo_root: Path,
    git_commit: str,
    doc_version: str,
) -> list[Chunk]:
    """Split one markdown file into Chunks, preserving heading hierarchy as provenance."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _strip_frontmatter(raw)
    source_file = path.relative_to(repo_root).as_posix()

    matches = list(_HEADING_RE.finditer(text))
    chunks: list[Chunk] = []
    heading_path: list[str] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        heading_path = _update_heading_path(heading_path, level, heading)

        # Body is everything between this heading and the next (or end of file)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        if not body:
            continue

        if _token_count(body) > _MAX_TOKENS:
            logger.debug("skipping oversized chunk '%s' in %s", heading, source_file)
            continue

        chunks.append(
            Chunk(
                id=f"{source_file}::{len(chunks)}",
                text=body,
                source_file=source_file,
                git_commit=git_commit,
                doc_version=doc_version,
                chunk_index=len(chunks),
                heading_path=list(heading_path),
            )
        )

    # Files with no headings: treat the whole body as a single chunk
    if not matches:
        body = text.strip()
        if body and _token_count(body) <= _MAX_TOKENS:
            chunks.append(
                Chunk(
                    id=f"{source_file}::0",
                    text=body,
                    source_file=source_file,
                    git_commit=git_commit,
                    doc_version=doc_version,
                    chunk_index=0,
                    heading_path=[],
                )
            )

    logger.debug("chunked %s → %d chunks", source_file, len(chunks))
    return chunks


def chunk_docs(
    paths: list[Path],
    repo_root: Path,
    git_commit: str,
    doc_version: str,
) -> list[Chunk]:
    """Chunk all doc files. Returns a flat list of Chunks across all files."""
    all_chunks: list[Chunk] = []
    for path in paths:
        all_chunks.extend(chunk_file(path, repo_root, git_commit, doc_version))
    logger.info("chunked %d files → %d total chunks", len(paths), len(all_chunks))
    return all_chunks
