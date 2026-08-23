from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vidore_rag.artifacts import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from vidore_rag.contracts import EvaluationCapabilities, JudgmentRecord, QueryRecord
from vidore_rag.datasets.base import DatasetAdapter
from vidore_rag.document_ir import PageRecord
from vidore_rag.ingestion.fingerprints import stage_fingerprint


class MaterializedPage(BaseModel):
    page: PageRecord
    image_path: str
    image_sha256: str = Field(min_length=64, max_length=64)
    image_format: str


class MaterializationManifest(BaseModel):
    schema_version: str = "1"
    dataset_id: str
    revision: str
    split: str
    language: str
    fingerprint: str = Field(min_length=64, max_length=64)
    page_count: int = Field(ge=1)
    query_count: int = Field(ge=0)
    judgment_count: int = Field(ge=0)
    invalid_bounding_box_count: int = Field(ge=0)
    capabilities: EvaluationCapabilities | None = None
    pages_path: str
    queries_path: str
    judgments_path: str


class MaterializationResult(BaseModel):
    manifest_path: str
    manifest: MaterializationManifest
    reused: bool


def materialize_dataset(
    adapter: DatasetAdapter,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> MaterializationResult:
    descriptor = adapter.descriptor
    fingerprint = stage_fingerprint(
        stage_name="materialize",
        implementation_version="2",
        config={
            "dataset_id": descriptor.dataset_id,
            "revision": descriptor.revision,
            "split": descriptor.split,
            "language": descriptor.language,
            "image_encoding": "jpeg-source-or-quality-95",
            "limit": limit,
        },
    )
    stage_dir = output_dir / fingerprint
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        existing = MaterializationManifest.model_validate(read_json(manifest_path))
        if existing.fingerprint == fingerprint and validate_materialization(
            manifest_path, verify_hashes=True
        ):
            return MaterializationResult(
                manifest_path=str(manifest_path), manifest=existing, reused=True
            )

    stage_dir.mkdir(parents=True, exist_ok=True)
    image_dir = stage_dir / "pages"
    image_dir.mkdir(parents=True, exist_ok=True)

    pages: list[MaterializedPage] = []
    for index, asset in enumerate(adapter.iter_page_assets()):
        if limit is not None and index >= limit:
            break
        image_path = image_dir / f"{asset.source_key}.jpg"
        _write_image(asset.image, image_path)
        materialized_page = asset.page.model_copy(update={"image_uri": str(image_path)})
        pages.append(
            MaterializedPage(
                page=materialized_page,
                image_path=str(image_path.relative_to(stage_dir)),
                image_sha256=sha256_file(image_path),
                image_format="jpeg",
            )
        )

    selected_page_ids = {page.page.page_id for page in pages}
    queries = list(adapter.iter_queries())
    judgments = [
        judgment
        for judgment in adapter.iter_judgments()
        if judgment.target_id in selected_page_ids
    ]
    selected_query_ids = {judgment.query_id for judgment in judgments}
    queries = [query for query in queries if query.query_id in selected_query_ids]

    pages_path = stage_dir / "pages.jsonl"
    queries_path = stage_dir / "queries.jsonl"
    judgments_path = stage_dir / "judgments.jsonl"
    write_jsonl(pages_path, (page.model_dump(mode="json") for page in pages))
    write_jsonl(queries_path, (query.model_dump(mode="json") for query in queries))
    write_jsonl(
        judgments_path, (judgment.model_dump(mode="json") for judgment in judgments)
    )

    manifest = MaterializationManifest(
        dataset_id=descriptor.dataset_id,
        revision=descriptor.revision,
        split=descriptor.split,
        language=descriptor.language,
        fingerprint=fingerprint,
        page_count=len(pages),
        query_count=len(queries),
        judgment_count=len(judgments),
        invalid_bounding_box_count=sum(
            judgment.invalid_bounding_box_count for judgment in judgments
        ),
        capabilities=adapter.capabilities(),
        pages_path=pages_path.name,
        queries_path=queries_path.name,
        judgments_path=judgments_path.name,
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))
    if not validate_materialization(manifest_path, verify_hashes=True):
        raise RuntimeError("materialized corpus failed validation")
    return MaterializationResult(
        manifest_path=str(manifest_path), manifest=manifest, reused=False
    )


def load_materialized_pages(manifest_path: Path) -> list[MaterializedPage]:
    manifest = MaterializationManifest.model_validate(read_json(manifest_path))
    rows = read_jsonl(manifest_path.parent / manifest.pages_path)
    return [MaterializedPage.model_validate(row) for row in rows]


def load_queries(manifest_path: Path) -> list[QueryRecord]:
    manifest = MaterializationManifest.model_validate(read_json(manifest_path))
    rows = read_jsonl(manifest_path.parent / manifest.queries_path)
    return [QueryRecord.model_validate(row) for row in rows]


def load_judgments(manifest_path: Path) -> list[JudgmentRecord]:
    manifest = MaterializationManifest.model_validate(read_json(manifest_path))
    rows = read_jsonl(manifest_path.parent / manifest.judgments_path)
    return [JudgmentRecord.model_validate(row) for row in rows]


def validate_materialization(manifest_path: Path, *, verify_hashes: bool) -> bool:
    try:
        manifest = MaterializationManifest.model_validate(read_json(manifest_path))
        pages = load_materialized_pages(manifest_path)
        queries = load_queries(manifest_path)
        judgments = load_judgments(manifest_path)
    except (OSError, ValueError):
        return False

    if len(pages) != manifest.page_count:
        return False
    if len(queries) != manifest.query_count or len(judgments) != manifest.judgment_count:
        return False
    if len({page.page.page_id for page in pages}) != len(pages):
        return False
    if len({query.query_id for query in queries}) != len(queries):
        return False

    query_ids = {query.query_id for query in queries}
    page_ids = {page.page.page_id for page in pages}
    if any(
        judgment.query_id not in query_ids or judgment.target_id not in page_ids
        for judgment in judgments
    ):
        return False

    for page in pages:
        image_path = manifest_path.parent / page.image_path
        if not image_path.is_file():
            return False
        if verify_hashes and sha256_file(image_path) != page.image_sha256:
            return False
    return True


def _write_image(image: Any, path: Path) -> None:
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.tmp")
    if not hasattr(image, "save"):
        raise TypeError("dataset image is not a decodable image object")
    converted = image.convert("RGB") if getattr(image, "mode", "RGB") != "RGB" else image
    converted.save(temporary, format="JPEG", quality=95, optimize=True)
    os.replace(temporary, path)
