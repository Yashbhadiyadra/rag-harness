"""In-memory BM25 index over the ingested corpus.

BM25 is a sparse, keyword-based ranker. It complements dense retrieval by
catching exact-term queries — commands, flag names, resource types — that
semantic embeddings often miss on a technical corpus like the K8s docs.

The index is built once at retriever startup by loading all documents from
ChromaDB. Rebuilding is cheap: ~50k K8s chunks tokenise and index in under
two seconds on a laptop, so this is preferred over persisting a separate
BM25 file.
"""

import json
import logging
import re

import chromadb
from rank_bm25 import BM25Okapi

from rag_harness.config import settings
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens.

    Deliberately simple: no stemming, no stopword removal. Kubernetes docs
    contain many single-token identifiers ('kubectl', 'PodDisruptionBudget')
    that stemmers mangle and stopwords do not overlap with.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    """In-memory BM25Okapi index over all chunks in the ChromaDB collection."""

    def __init__(self) -> None:
        """Load all chunks from ChromaDB and build the BM25 index."""
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_collection(name=settings.chroma_collection)

        raw = collection.get(include=["documents", "metadatas"])
        documents = raw["documents"] or []
        metadatas = raw["metadatas"] or []
        ids = raw["ids"] or []

        self._chunks: list[Chunk] = []
        tokenised_corpus: list[list[str]] = []

        for chunk_id, doc, meta in zip(ids, documents, metadatas):
            if meta is None:
                continue
            self._chunks.append(
                Chunk(
                    id=chunk_id,
                    text=doc,
                    source_file=str(meta["source_file"]),
                    git_commit=str(meta["git_commit"]),
                    doc_version=str(meta["doc_version"]),
                    chunk_index=int(str(meta["chunk_index"])),
                    heading_path=json.loads(str(meta.get("heading_path", "[]"))),
                )
            )
            tokenised_corpus.append(tokenize(doc))

        if not tokenised_corpus:
            raise ValueError("BM25Store: collection is empty — run ingest first.")

        self._bm25 = BM25Okapi(tokenised_corpus)
        logger.info("BM25Store loaded %d chunks", len(self._chunks))

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks by BM25 score, best first.

        Score is the raw BM25 value — useful only for relative ranking within
        this store; do not compare across queries or across stores.
        """
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in ranked]

    def __len__(self) -> int:
        return len(self._chunks)
