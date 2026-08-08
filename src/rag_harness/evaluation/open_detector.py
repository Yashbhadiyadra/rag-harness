"""Open-model claim groundedness via a small NLI cross-encoder (ADR-0030).

A cheap, local, zero-API-cost alternative to the LLM claim judge (ADR-0027).
Each answer sentence is treated as a hypothesis and tested against the retrieved
context with a natural-language-inference cross-encoder: entailment means
grounded, contradiction means contradicted, neutral means ungrounded. This is
the Luna / LettuceDetect pattern (a compact encoder localizing unsupported
content) adapted to the harness, reusing the transformers/torch already declared
in the ``[rerank]`` extra, so it adds no new dependency.

NLI has no "complementary" class, so a reasonable-but-beyond-context claim is
labelled ungrounded (the conservative choice). The LLM judge remains the scorer
that can make that finer distinction; this is the cheap open second opinion.
"""

import logging
from typing import Any

from rag_harness.config import settings
from rag_harness.evaluation.claim_eval import CONTRADICTED, GROUNDED, UNGROUNDED, ClaimLabel
from rag_harness.models import Chunk

logger = logging.getLogger(__name__)

# NLI cross-encoder label order (verified for cross-encoder/nli-* models):
# 0 = contradiction, 1 = entailment, 2 = neutral.
_CONTRADICTION, _ENTAILMENT, _NEUTRAL = 0, 1, 2

_model: Any = None


def _get_model() -> Any:
    """Load and cache the NLI cross-encoder.

    Raises ImportError with install guidance if the rerank extra (which provides
    sentence-transformers) is absent.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the open detector needs sentence-transformers: pip install -e '.[rerank]'"
            ) from e
        logger.info("loading open detector model %s", settings.open_detector_model)
        _model = CrossEncoder(settings.open_detector_model)
    return _model


def _label_for(predictions: Any) -> str:
    """Map per-chunk NLI predictions for one claim to a single groundedness label.

    A claim entailed by any chunk is grounded (supported by some source);
    otherwise, contradicted by any chunk is contradicted; otherwise ungrounded.
    """
    import numpy as np

    argmax = np.asarray(predictions).argmax(axis=1)
    if (argmax == _ENTAILMENT).any():
        return GROUNDED
    if (argmax == _CONTRADICTION).any():
        return CONTRADICTED
    return UNGROUNDED


def classify_claims_open(answer: str, chunks: list[Chunk]) -> list[ClaimLabel]:
    """Label each answer sentence grounded/contradicted/ungrounded via NLI.

    No LLM call and no network: the cross-encoder runs locally. Returns an empty
    list when there is nothing to check (no answer sentences or no context).
    """
    from rag_harness.evaluation.citation_eval import sentences_with_markers

    sentences = [s for s, _ in sentences_with_markers(answer)]
    if not sentences or not chunks:
        return []

    model = _get_model()
    labels: list[ClaimLabel] = []
    for sentence in sentences:
        # premise = context chunk, hypothesis = the claim sentence
        pairs = [(chunk.text, sentence) for chunk in chunks]
        predictions = model.predict(pairs)
        labels.append(ClaimLabel(claim=sentence, label=_label_for(predictions)))
    return labels
