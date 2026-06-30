from abc import ABC, abstractmethod

from rag_harness.models import Chunk


class Retriever(ABC):
    """Abstract retrieval interface. All retrieval implementations must satisfy this contract.

    Keeping retrieval behind an interface means the generation and evaluation layers
    never depend on ChromaDB directly — swapping to hybrid or reranked retrieval
    requires no changes outside this module.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Return the top_k most relevant Chunks for the given query."""
        ...
