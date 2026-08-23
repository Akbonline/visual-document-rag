import ast
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vidore_rag.context import ContextBuilder
from vidore_rag.contracts import (
    AnswerFormat,
    EvaluationCapabilities,
    JudgmentCardinality,
    JudgmentRecord,
    JudgmentScale,
    JudgmentStructure,
    JudgmentUnit,
    QueryRecord,
)
from vidore_rag.datasets import DatasetAdapter, DatasetDescriptor, DatasetRegistry, PageAsset
from vidore_rag.document_ir import PageRecord
from vidore_rag.evaluation import BenchmarkSession, MetricStatus
from vidore_rag.ingestion import materialize_dataset
from vidore_rag.retrieval import build_text_index

ROOT = Path(__file__).resolve().parents[1]


class FakeImage:
    mode = "RGB"
    size = (120, 180)

    def save(self, path: Path, **_: Any) -> None:
        path.write_bytes(b"portable-fixture-image")


class SecondDatasetAdapter(DatasetAdapter):
    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_id="open/maintenance-manual",
            revision="fixture-v1",
            split="test",
            language="english",
        )

    def capabilities(self) -> EvaluationCapabilities:
        return EvaluationCapabilities(
            judgment_unit=JudgmentUnit.PAGE,
            judgment_cardinality=JudgmentCardinality.SINGLE,
            judgment_scale=JudgmentScale.GRADED,
            relevance_grades=[0, 1, 2],
            binary_relevance_threshold=1,
            judgment_structure=JudgmentStructure.SINGLE,
            has_reference_answers=True,
            answer_format=AnswerFormat.FREE_TEXT,
            has_answerability_label=False,
            has_bounding_box_evidence=False,
        )

    def iter_pages(self) -> Iterable[PageRecord]:
        return iter(self._pages())

    def iter_page_assets(self) -> Iterable[PageAsset]:
        for index, page in enumerate(self._pages()):
            yield PageAsset(source_key=f"page-{index}", page=page, image=FakeImage())

    def iter_queries(self) -> Iterable[QueryRecord]:
        yield QueryRecord(
            query_id="maintenance:q:42",
            text="What does ERR-42 mean?",
            is_answerable=None,
            reference_answers=["The battery temperature is unsafe."],
            metadata={"native_query_id": 42},
        )
        yield QueryRecord(
            query_id="maintenance:q:7",
            text="Which port carries telemetry?",
            is_answerable=None,
            reference_answers=["Port 7443 carries telemetry."],
            metadata={"native_query_id": 7},
        )

    def iter_judgments(self) -> Iterable[JudgmentRecord]:
        yield JudgmentRecord(
            judgment_id="maintenance:j:42",
            query_id="maintenance:q:42",
            target_id="maintenance:p:1",
            judgment_unit=JudgmentUnit.PAGE,
            relevance=2,
        )
        yield JudgmentRecord(
            judgment_id="maintenance:j:7",
            query_id="maintenance:q:7",
            target_id="maintenance:p:3",
            judgment_unit=JudgmentUnit.PAGE,
            relevance=2,
        )

    @staticmethod
    def _pages() -> list[PageRecord]:
        texts = [
            "General robot startup and shutdown procedures.",
            "ERR-42 means the battery temperature is above the safe operating range.",
            "Wheel calibration requires a level floor.",
            "Robot telemetry is transmitted over secure port 7443.",
            "Clean the camera lens with a microfiber cloth.",
        ]
        return [
            PageRecord(
                page_id=f"maintenance:p:{index}",
                document_id="maintenance:manual",
                page_number=index + 1,
                text=text,
                width=120,
                height=180,
            )
            for index, text in enumerate(texts)
        ]


def test_second_dataset_reaches_scored_cited_page_evidence(tmp_path: Path) -> None:
    materialized = materialize_dataset(SecondDatasetAdapter(), tmp_path / "materialized")
    indexed = build_text_index(
        Path(materialized.manifest_path),
        tmp_path / "indexes",
        policy="fixed",
        chunk_size=32,
        overlap=4,
    )
    session = BenchmarkSession(Path(indexed.manifest_path), mode="bm25")

    report = session.evaluate()
    result = session.query_by_native_id(42)
    context = ContextBuilder(token_budget=64).build(result.hits)

    assert materialized.manifest.page_count == 5
    assert indexed.manifest.chunk_count == 5
    assert indexed.manifest.projection is not None
    assert len(indexed.manifest.projection.projection_fingerprint) == 64
    assert report.aggregate.scored_query_count == 2
    assert result.metrics.status is MetricStatus.APPLICABLE
    assert result.hits[0].page_id == "maintenance:p:1"
    assert result.gold_pages[result.hits[0].page_id] == 2
    assert "maintenance:manual page 1" in context.rendered_context


def test_registry_constructs_the_second_adapter_from_generic_config() -> None:
    registry = DatasetRegistry()
    registry.register("second", lambda _: SecondDatasetAdapter())

    assert registry.names() == ("second",)


def test_core_modules_do_not_import_a_dataset_specific_adapter() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "src/vidore_rag").rglob("*.py")):
        if "datasets" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        if any(module.endswith("datasets.vidore_v3") for module in imports):
            violations.append(str(path.relative_to(ROOT)))

    assert violations == []
