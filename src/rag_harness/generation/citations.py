"""Parse inline chunk-level citations from a grounded answer.

The generator numbers context passages [1], [2], ... and is instructed to
cite the passages it used inline with those markers. This module extracts
which passage indices the answer actually cited, so the API can attribute
each claim to a specific retrieved chunk (chunk-level, not just file-level,
attribution - the NVIDIA NeMo Retriever standard).

Pure and network-free: it operates on the answer string plus the chunk
list the answer was generated from.
"""

import re

from rag_harness.models import Chunk

# Matches a citation marker like [1] or [12]. Deliberately narrow: a bare
# number in prose ("port 8080") is not a marker, only bracketed integers.
_MARKER = re.compile(r"\[(\d+)\]")


def cited_chunk_indices(answer: str) -> list[int]:
    """Return the 1-based passage indices cited in *answer*, sorted, unique."""
    seen = {int(m.group(1)) for m in _MARKER.finditer(answer)}
    return sorted(seen)


def resolve_citations(answer: str, chunks: list[Chunk]) -> list[Chunk]:
    """Return the chunks the answer cited, in citation order.

    Indices are 1-based and map to ``chunks[index - 1]``. Out-of-range
    markers (a model citing a passage number that does not exist) are
    skipped rather than raising - a stray marker must not break a response.
    """
    resolved: list[Chunk] = []
    for index in cited_chunk_indices(answer):
        if 1 <= index <= len(chunks):
            resolved.append(chunks[index - 1])
    return resolved
