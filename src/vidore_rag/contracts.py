from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class JudgmentUnit(StrEnum):
    DOCUMENT = "document"
    PAGE = "page"
    REGION = "region"


class RetrievalUnit(StrEnum):
    CHUNK = "chunk"
    PAGE = "page"
    DOCUMENT = "document"
    REGION = "region"


class JudgmentCardinality(StrEnum):
    NONE = "none"
    SINGLE = "single"
    MULTIPLE = "multiple"
    VARIABLE = "variable"


class JudgmentScale(StrEnum):
    BINARY = "binary"
    GRADED = "graded"


class JudgmentStructure(StrEnum):
    UNKNOWN = "unknown"
    SINGLE = "single"
    ANY_OF = "any_of"
    ALL_OF = "all_of"
    GROUPED = "grouped"


class AnswerFormat(StrEnum):
    NONE = "none"
    FREE_TEXT = "free_text"
    MULTIPLE_CHOICE = "multiple_choice"
    STRUCTURED = "structured"


class Aggregation(StrEnum):
    BEST_RANK = "best_rank"
    MAX_SCORE = "max_score"
    SUM_SCORE = "sum_score"
    RRF = "rrf"


class EvaluationCapabilities(BaseModel):
    judgment_unit: JudgmentUnit
    judgment_cardinality: JudgmentCardinality
    judgment_scale: JudgmentScale
    relevance_grades: list[float] = Field(min_length=1)
    binary_relevance_threshold: float | None = None
    judgment_structure: JudgmentStructure = JudgmentStructure.UNKNOWN
    has_reference_answers: bool
    answer_format: AnswerFormat
    has_answerability_label: bool
    has_bounding_box_evidence: bool

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        grades = sorted(set(self.relevance_grades))
        if self.judgment_scale is JudgmentScale.GRADED and len(grades) < 2:
            raise ValueError("graded judgments require at least two relevance grades")

        if self.binary_relevance_threshold is None:
            raise ValueError("retrieval evaluation requires a binary relevance threshold")
        if self.binary_relevance_threshold > max(grades):
            raise ValueError("binary relevance threshold exceeds every declared grade")

        if self.has_reference_answers and self.answer_format is AnswerFormat.NONE:
            raise ValueError("reference answers require a non-none answer format")
        if not self.has_reference_answers and self.answer_format is not AnswerFormat.NONE:
            raise ValueError("answer format must be none when reference answers are unavailable")
        return self


class DatasetSelection(BaseModel):
    adapter: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    query_language: str | None = None
    max_pages: int = Field(ge=1, le=2000)
    deterministic_seed: int
    license_id: str = Field(min_length=1)
    gated: bool


class EvaluationConfig(BaseModel):
    common_unit: JudgmentUnit
    require_reference_answers_for_generation_metrics: bool = True
    refuse_unknown_projection: bool = True
    refuse_unknown_gold_semantics: bool = True
    capabilities: EvaluationCapabilities

    @model_validator(mode="after")
    def validate_common_unit(self) -> Self:
        if self.common_unit is not self.capabilities.judgment_unit:
            raise ValueError("common evaluation unit must match the dataset judgment unit")
        return self


class ProjectDatasetConfig(BaseModel):
    dataset: DatasetSelection
    evaluation: EvaluationConfig


class EvidenceRequirementGroup(BaseModel):
    group_id: str = Field(min_length=1)
    mode: Literal["any_of", "all_of"]
    judgment_ids: list[str] = Field(min_length=1)


class QueryRecord(BaseModel):
    query_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    is_answerable: bool | None
    reference_answers: list[str] = Field(default_factory=list)
    requirement_groups: list[EvidenceRequirementGroup] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BoundingBoxEvidence(BaseModel):
    coordinate_space: Literal["pixel"] = "pixel"
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)
    annotator: int | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box maximums must exceed minimums")
        return self


class JudgmentRecord(BaseModel):
    judgment_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    judgment_unit: JudgmentUnit
    relevance: float
    content_types: list[str] = Field(default_factory=list)
    bounding_boxes: list[BoundingBoxEvidence] = Field(default_factory=list)
    invalid_bounding_box_count: int = Field(default=0, ge=0)


class IndexManifest(BaseModel):
    index_id: str = Field(min_length=1)
    retrieval_unit: RetrievalUnit
    corpus_fingerprint: str = Field(min_length=64, max_length=64)
    index_fingerprint: str = Field(min_length=64, max_length=64)
    retriever_name: str = Field(min_length=1)
    retriever_revision: str = Field(min_length=1)


class UnitProjection(BaseModel):
    source_unit: RetrievalUnit
    target_unit: JudgmentUnit
    mapping_artifact_uri: str = Field(min_length=1)
    aggregation: Aggregation
    projection_fingerprint: str = Field(min_length=64, max_length=64)


def validate_evaluation_units(
    index: IndexManifest,
    capabilities: EvaluationCapabilities,
    projection: UnitProjection | None,
) -> None:
    if index.retrieval_unit.value == capabilities.judgment_unit.value:
        return
    if projection is None:
        raise ValueError(
            f"retrieval unit {index.retrieval_unit.value!r} cannot be scored against "
            f"judgment unit {capabilities.judgment_unit.value!r} without a projection"
        )
    if projection.source_unit is not index.retrieval_unit:
        raise ValueError("projection source unit does not match the index retrieval unit")
    if projection.target_unit is not capabilities.judgment_unit:
        raise ValueError("projection target unit does not match the dataset judgment unit")
