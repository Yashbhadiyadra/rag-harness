"""Factuality gateway: verify an answer's claims and regenerate the bad ones (ADR-0029).

After an answer is drafted, classify its atomic claims against the retrieved
context (reusing ADR-0027). If any claim is ungrounded or contradicted,
regenerate once with the flagged claims fed back and an instruction to answer
using only context-supported content. When nothing is supported this collapses
to the standard refusal, which is the correct outcome.

Gated by ``settings.factuality_gateway_enabled`` (default off): it adds LLM calls
and earns its keep on low-trust corpora, not on the already-grounded pinned demo
corpus.
"""

import logging
from dataclasses import dataclass

from rag_harness.config import settings
from rag_harness.evaluation.claim_eval import CONTRADICTED, UNGROUNDED, classify_claims
from rag_harness.generation.generator import _FALLBACK_ANSWER, _build_context
from rag_harness.models import Chunk
from rag_harness.observability.usage import TokenUsage, record_usage
from rag_harness.openai_client import build_async_client

logger = logging.getLogger(__name__)

_client = build_async_client()

_REVISE_SYSTEM_PROMPT = """\
You are revising a draft answer about Kubernetes. Some claims in the draft are \
not supported by the provided context and are listed as UNSUPPORTED CLAIMS. \
Rewrite the answer using ONLY information in the context passages, and remove \
every unsupported claim. Do not add new outside knowledge. If, after removing \
the unsupported claims, the context does not support any answer to the question, \
reply exactly:
"I do not have enough information in the provided context to answer this question."

Cite passages inline like [1] exactly as the original instructions require.
"""


@dataclass
class GatewayResult:
    """Outcome of one factuality-gateway pass."""

    answer: str
    revised: bool
    n_claims: int
    n_flagged: int  # ungrounded + contradicted claims in the draft


async def _regenerate(query: str, chunks: list[Chunk], flagged: list[str]) -> str:
    """Regenerate the answer once, omitting the flagged unsupported claims."""
    context = _build_context(chunks)
    flagged_block = "\n".join(f"- {c}" for c in flagged)
    user_message = (
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {query}\n\n"
        f"UNSUPPORTED CLAIMS to remove:\n{flagged_block}"
    )
    response = await _client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": _REVISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    record_usage(TokenUsage.from_openai(settings.generation_model, response))
    return response.choices[0].message.content or _FALLBACK_ANSWER


async def factuality_gateway(query: str, answer: str, chunks: list[Chunk]) -> GatewayResult:
    """Verify the answer's claims; regenerate once if any are unsupported."""
    labels = await classify_claims(query, answer, chunks)
    flagged = [cl.claim for cl in labels if cl.label in (UNGROUNDED, CONTRADICTED)]
    if not flagged:
        return GatewayResult(answer=answer, revised=False, n_claims=len(labels), n_flagged=0)

    logger.info(
        "factuality gateway: %d/%d claims unsupported, regenerating", len(flagged), len(labels)
    )
    revised = await _regenerate(query, chunks, flagged)
    return GatewayResult(answer=revised, revised=True, n_claims=len(labels), n_flagged=len(flagged))
