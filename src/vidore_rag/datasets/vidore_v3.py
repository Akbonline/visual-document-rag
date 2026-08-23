from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from typing import Any, ClassVar

from pydantic import ValidationError

from vidore_rag.contracts import (
    AnswerFormat,
    BoundingBoxEvidence,
    EvaluationCapabilities,
    JudgmentCardinality,
    JudgmentRecord,
    JudgmentScale,
    JudgmentStructure,
    JudgmentUnit,
    QueryRecord,
)
from vidore_rag.datasets.base import DatasetAdapter, DatasetDescriptor, PageAsset
from vidore_rag.document_ir import PageRecord


class ViDoReV3Adapter(DatasetAdapter):
    """Translate the frozen ViDoRe V3 HR schema into canonical records."""

    DATASET_ID: ClassVar[str] = "vidore/vidore_v3_hr"
    REVISION: ClassVar[str] = "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3"
    SPLIT: ClassVar[str] = "test"

    def __init__(
        self,
        *,
        corpus: Iterable[Mapping[str, Any]],
        queries: Iterable[Mapping[str, Any]],
        qrels: Iterable[Mapping[str, Any]],
        language: str = "english",
    ) -> None:
        self._corpus = tuple(corpus)
        self._queries = tuple(row for row in queries if row.get("language") == language)
        native_query_ids = {int(row["query_id"]) for row in self._queries}
        self._qrels = tuple(
            row for row in qrels if int(row["query_id"]) in native_query_ids
        )
        self._language = language

    @property
    def language(self) -> str:
        return self._language

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_id=self.DATASET_ID,
            revision=self.REVISION,
            split=self.SPLIT,
            language=self.language,
        )

    @classmethod
    def from_huggingface(cls, *, language: str = "english") -> ViDoReV3Adapter:
        try:
            load_dataset = import_module("datasets").load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "install the prototype dependencies to load Hugging Face data"
            ) from exc

        options = {"revision": cls.REVISION, "split": cls.SPLIT}
        return cls(
            corpus=load_dataset(cls.DATASET_ID, "corpus", **options),
            queries=load_dataset(cls.DATASET_ID, "queries", **options),
            qrels=load_dataset(cls.DATASET_ID, "qrels", **options),
            language=language,
        )

    def capabilities(self) -> EvaluationCapabilities:
        return EvaluationCapabilities(
            judgment_unit=JudgmentUnit.PAGE,
            judgment_cardinality=JudgmentCardinality.VARIABLE,
            judgment_scale=JudgmentScale.GRADED,
            relevance_grades=[0, 1, 2],
            binary_relevance_threshold=1,
            judgment_structure=JudgmentStructure.UNKNOWN,
            has_reference_answers=True,
            answer_format=AnswerFormat.FREE_TEXT,
            has_answerability_label=False,
            has_bounding_box_evidence=True,
        )

    def iter_pages(self) -> Iterable[PageRecord]:
        for row in self._corpus:
            yield self._page_from_row(row)

    def iter_page_assets(self) -> Iterable[PageAsset]:
        for row in self._corpus:
            native_page_id = int(row["corpus_id"])
            yield PageAsset(
                source_key=f"{native_page_id:06d}",
                page=self._page_from_row(row),
                image=row.get("image"),
            )

    def iter_queries(self) -> Iterable[QueryRecord]:
        for row in self._queries:
            native_query_id = int(row["query_id"])
            answer = str(row["answer"]).strip()
            if not answer:
                raise ValueError(f"query {native_query_id} has an empty reference answer")
            yield QueryRecord(
                query_id=self._query_id(native_query_id),
                text=str(row["query"]),
                is_answerable=None,
                reference_answers=[answer],
                metadata={
                    "native_query_id": native_query_id,
                    "language": self._language,
                    "query_types": _string_list(row.get("query_types")),
                    "content_type": _string_list(row.get("content_type")),
                },
            )

    def iter_judgments(self) -> Iterable[JudgmentRecord]:
        for row in self._qrels:
            native_query_id = int(row["query_id"])
            native_page_id = int(row["corpus_id"])
            boxes, invalid_box_count = _parse_bounding_boxes(row["bounding_boxes"])
            yield JudgmentRecord(
                judgment_id=(
                    f"{self.DATASET_ID}:judgment:{native_query_id}:{native_page_id}"
                ),
                query_id=self._query_id(native_query_id),
                target_id=self._page_id(native_page_id),
                judgment_unit=JudgmentUnit.PAGE,
                relevance=float(row["score"]),
                content_types=_string_list(row.get("content_type")),
                bounding_boxes=boxes,
                invalid_bounding_box_count=invalid_box_count,
            )

    @classmethod
    def _page_id(cls, native_page_id: int) -> str:
        return f"{cls.DATASET_ID}:page:{native_page_id}"

    @classmethod
    def _query_id(cls, native_query_id: int) -> str:
        return f"{cls.DATASET_ID}:query:{native_query_id}"

    @classmethod
    def _page_from_row(cls, row: Mapping[str, Any]) -> PageRecord:
        native_page_id = int(row["corpus_id"])
        width, height = _image_dimensions(row.get("image"))
        return PageRecord(
            page_id=cls._page_id(native_page_id),
            document_id=f"{cls.DATASET_ID}:document:{row['doc_id']}",
            page_number=int(row["page_number_in_doc"]) + 1,
            text=str(row.get("markdown") or ""),
            image_uri=(
                f"hf://{cls.DATASET_ID}@{cls.REVISION}/corpus/{native_page_id}/image"
            ),
            width=width,
            height=height,
            metadata={
                "native_corpus_id": native_page_id,
                "native_page_number": int(row["page_number_in_doc"]),
                "text_source": "shipped_markdown",
            },
        )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in value]


def _image_dimensions(image: Any) -> tuple[int | None, int | None]:
    if image is None:
        return None, None
    if isinstance(image, Mapping):
        width = image.get("width")
        height = image.get("height")
        return (
            int(width) if width is not None else None,
            int(height) if height is not None else None,
        )
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    return None, None


def _parse_bounding_boxes(
    values: Iterable[Mapping[str, Any]],
) -> tuple[list[BoundingBoxEvidence], int]:
    valid: list[BoundingBoxEvidence] = []
    invalid_count = 0
    for value in values:
        try:
            valid.append(BoundingBoxEvidence.model_validate(dict(value)))
        except ValidationError:
            invalid_count += 1
    return valid, invalid_count
