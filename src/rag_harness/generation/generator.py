import logging

from openai import OpenAI

from rag_harness.config import settings
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a technical assistant that answers questions about Kubernetes.
Answer using ONLY the information in the provided context passages.
If the context does not contain enough information to answer, say:
"I don't have enough information in the provided context to answer this question."
Do not use any outside knowledge. Be concise and precise.
"""


def _build_context(chunks: list[Chunk]) -> str:
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.source_file
        sections.append(f"[{i}] {heading}\n{chunk.text}")
    return "\n\n---\n\n".join(sections)


def generate(query: str, chunks: list[Chunk]) -> str:
    """Generate a grounded answer from retrieved chunks.

    The prompt is structured so the model is explicitly constrained to the
    provided context — this is what faithfulness scoring measures against.
    """
    if not chunks:
        return "I do not have enough information in the provided context to answer this question."

    context = _build_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {query}"

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,  # deterministic output — important for reproducible eval
    )

    answer = response.choices[0].message.content or ""
    logger.debug("generated answer (%d chars) for query: %.60s...", len(answer), query)
    return answer
