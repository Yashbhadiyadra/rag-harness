"""Dense vector retrieval: embed query, search ChromaDB, reconstruct typed Chunks."""

import asyncio
import json
import logging

import chromadb

from rag_harness.config import settings
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage
from rag_harness.openai_client import build_async_client
from rag_harness.retrieval.base import Retriever

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """Retrieves chunks using dense vector similarity search against ChromaDB."""

    def __init__(self, collection_name: str | None = None) -> None:
        """Open the given Chroma collection (default ``settings.chroma_collection``).

        ``collection_name`` is the per-tenant isolation boundary (ADR-0025): a
        retriever bound to tenant A's collection can only ever return A's chunks.
        """
        self._openai = build_async_client()
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self._collection = client.get_collection(name=collection_name or settings.chroma_collection)

    async def _embed_query(self, query: str) -> list[float]:
        response = await self._openai.embeddings.create(
            model=settings.embedding_model,
            input=[query],
        )
        record_usage(TokenUsage.from_openai(settings.embedding_model, response))
        return response.data[0].embedding

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Embed the query and return the top_k most similar chunks from ChromaDB."""
        k = top_k if top_k is not None else settings.retrieval_top_k
        query_vector = await self._embed_query(query)

        # ChromaDB's query() is sync/blocking. Wrap in to_thread so we don't
        # block the event loop for concurrent /query requests.
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_vector],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas"],
        )

        chunks: list[Chunk] = []
        documents = results["documents"] or [[]]
        metadatas = results["metadatas"] or [[]]
        ids = results["ids"] or [[]]

        for doc, meta, chunk_id in zip(documents[0], metadatas[0], ids[0]):
            if meta is None:
                continue
            chunks.append(
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

        logger.debug("retrieved %d chunks for query: %.60s...", len(chunks), query)
        return chunks
