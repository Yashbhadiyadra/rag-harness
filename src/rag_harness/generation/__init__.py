"""Generation layer: retrieved context + query → grounded answer via gpt-4o-mini."""

from rag_harness.generation.corrective import CorrectiveResult, corrective_generate
from rag_harness.generation.critic import Category, RelevanceCritic
from rag_harness.generation.generator import generate

__all__ = [
    "Category",
    "CorrectiveResult",
    "RelevanceCritic",
    "corrective_generate",
    "generate",
]
