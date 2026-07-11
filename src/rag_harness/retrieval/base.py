"""Abstract retrieval interface shared by all retriever implementations."""

import asyncio
from abc import ABC, abstractmethod

from rag_harness.models import Chunk


class Retriever(ABC):
    """Abstract retrieval interface. All retrieval implementations must satisfy this contract.

    Keeping retrieval behind an interface means the generation and evaluation layers
    never depend on ChromaDB directly - swapping to hybrid or reranked retrieval
    requires no changes outside this module.

    Async-first: implementations override ``retrieve_async``. ``retrieve`` is a
    sync facade kept for callers that have not migrated yet; it runs the async
    implementation in a fresh event loop via ``asyncio.run``.
    """

    @abstractmethod
    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return the top_k most relevant Chunks for the given query (async)."""
        ...

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Sync facade - runs the async implementation in a fresh event loop.

        Kept for the transition period only. New callers should ``await
        retrieve_async`` directly.
        """
        return asyncio.run(self.retrieve_async(query, top_k))
