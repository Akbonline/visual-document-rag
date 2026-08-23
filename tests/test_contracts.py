import pytest
from pydantic import ValidationError

from vidore_rag.contracts import (
    Aggregation,
    AnswerFormat,
    EvaluationCapabilities,
    IndexManifest,
    JudgmentCardinality,
    JudgmentScale,
    JudgmentStructure,
    JudgmentUnit,
    RetrievalUnit,
    UnitProjection,
    validate_evaluation_units,
)

HEX_A = "a" * 64
HEX_B = "b" * 64


def capabilities() -> EvaluationCapabilities:
    return EvaluationCapabilities(
        judgment_unit=JudgmentUnit.PAGE,
        judgment_cardinality=JudgmentCardinality.MULTIPLE,
        judgment_scale=JudgmentScale.GRADED,
        relevance_grades=[0, 1, 2],
        binary_relevance_threshold=1,
        judgment_structure=JudgmentStructure.UNKNOWN,
        has_reference_answers=True,
        answer_format=AnswerFormat.FREE_TEXT,
        has_answerability_label=False,
        has_bounding_box_evidence=True,
    )


def test_reference_answer_capability_requires_answer_format() -> None:
    with pytest.raises(ValidationError):
        EvaluationCapabilities(
            judgment_unit=JudgmentUnit.PAGE,
            judgment_cardinality=JudgmentCardinality.SINGLE,
            judgment_scale=JudgmentScale.BINARY,
            relevance_grades=[0, 1],
            binary_relevance_threshold=1,
            judgment_structure=JudgmentStructure.SINGLE,
            has_reference_answers=True,
            answer_format=AnswerFormat.NONE,
            has_answerability_label=False,
            has_bounding_box_evidence=False,
        )


def test_projection_is_required_for_chunk_index_against_page_gold() -> None:
    index = IndexManifest(
        index_id="chunks",
        retrieval_unit=RetrievalUnit.CHUNK,
        corpus_fingerprint=HEX_A,
        index_fingerprint=HEX_B,
        retriever_name="bm25",
        retriever_revision="fixture",
    )

    with pytest.raises(ValueError, match="without a projection"):
        validate_evaluation_units(index, capabilities(), projection=None)

    projection = UnitProjection(
        source_unit=RetrievalUnit.CHUNK,
        target_unit=JudgmentUnit.PAGE,
        mapping_artifact_uri="artifacts/chunk-to-page.json",
        aggregation=Aggregation.BEST_RANK,
        projection_fingerprint=HEX_A,
    )
    validate_evaluation_units(index, capabilities(), projection)

