"""Tests for the open-model NLI groundedness detector (ADR-0030).

The model is mocked so the suite never downloads weights; these tests cover the
label mapping and the per-sentence classification wiring.
"""

from unittest.mock import MagicMock, patch

from rag_harness.evaluation.claim_eval import CONTRADICTED, GROUNDED, UNGROUNDED
from rag_harness.evaluation.open_detector import _label_for, classify_claims_open
from rag_harness.models import Chunk

# NLI label columns: [contradiction, entailment, neutral]
_ENTAIL = [-1.0, 5.0, 0.0]
_CONTRA = [5.0, -1.0, 0.0]
_NEUTRAL = [0.0, -1.0, 5.0]


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="c::0",
        text=text,
        source_file="d.md",
        git_commit="abc",
        doc_version="v1",
        chunk_index=0,
        heading_path=[],
    )


def test_label_for_entailment_is_grounded() -> None:
    assert _label_for([_ENTAIL]) == GROUNDED
    # entailment by any chunk wins even if another contradicts
    assert _label_for([_CONTRA, _ENTAIL]) == GROUNDED


def test_label_for_contradiction_without_entailment() -> None:
    assert _label_for([_CONTRA]) == CONTRADICTED
    assert _label_for([_NEUTRAL, _CONTRA]) == CONTRADICTED


def test_label_for_neutral_is_ungrounded() -> None:
    assert _label_for([_NEUTRAL]) == UNGROUNDED
    assert _label_for([_NEUTRAL, _NEUTRAL]) == UNGROUNDED


def test_classify_claims_open_labels_each_sentence() -> None:
    model = MagicMock()
    # first sentence entailed, second contradicted (one chunk -> one row each)
    model.predict.side_effect = [[_ENTAIL], [_CONTRA]]
    with patch("rag_harness.evaluation.open_detector._get_model", return_value=model):
        labels = classify_claims_open(
            "A Pod wraps containers. Pods scale to zero by default.", [_chunk("ctx")]
        )

    assert [cl.label for cl in labels] == [GROUNDED, CONTRADICTED]


def test_classify_claims_open_empty_inputs() -> None:
    with patch("rag_harness.evaluation.open_detector._get_model") as get_model:
        assert classify_claims_open("", [_chunk("ctx")]) == []
        assert classify_claims_open("A claim.", []) == []
        get_model.assert_not_called()  # nothing to score, model never loaded
