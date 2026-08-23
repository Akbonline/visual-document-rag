from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vidore_rag.artifacts import read_json, read_jsonl, write_json, write_jsonl
from vidore_rag.chunking import FixedTokenChunker, StructureAwareChunker
from vidore_rag.document_ir import ChunkRecord, PageRecord
from vidore_rag.ingestion.fingerprints import stage_fingerprint
from vidore_rag.ingestion.materialize import (
    MaterializationManifest,
    load_materialized_pages,
)
from vidore_rag.ocr import OCRManifest, load_ocr_pages

ChunkPolicy = Literal["fixed", "structure"]
TextSource = Literal["markdown", "ocr"]


class TextIndexManifest(BaseModel):
    schema_version: str = "1"
    source_manifest_path: str
    source_fingerprint: str = Field(min_length=64, max_length=64)
    fingerprint: str = Field(min_length=64, max_length=64)
    text_source: TextSource
    ocr_manifest_path: str | None = None
    chunk_policy: ChunkPolicy
    chunk_size: int = Field(ge=1)
    overlap: int = Field(ge=0)
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    chunks_path: str = "chunks.jsonl"


class TextIndexResult(BaseModel):
    manifest_path: str
    manifest: TextIndexManifest
    reused: bool


def build_text_index(
    source_manifest_path: Path,
    output_root: Path,
    *,
    policy: ChunkPolicy = "fixed",
    text_source: TextSource = "markdown",
    ocr_manifest_path: Path | None = None,
    chunk_size: int = 180,
    overlap: int = 24,
) -> TextIndexResult:
    source_manifest = MaterializationManifest.model_validate(read_json(source_manifest_path))
    upstream = [source_manifest.fingerprint]
    if text_source == "ocr":
        if ocr_manifest_path is None:
            raise ValueError("OCR text source requires an OCR manifest")
        ocr_manifest = OCRManifest.model_validate(read_json(ocr_manifest_path))
        if ocr_manifest.source_fingerprint != source_manifest.fingerprint:
            raise ValueError("OCR and materialization fingerprints do not match")
        upstream.append(ocr_manifest.fingerprint)
    elif ocr_manifest_path is not None:
        raise ValueError("OCR manifest may only be set when text_source is ocr")

    fingerprint = stage_fingerprint(
        stage_name="text_index",
        implementation_version="2",
        config={
            "policy": policy,
            "text_source": text_source,
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        upstream_fingerprints=upstream,
    )
    stage_dir = output_root / fingerprint
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        manifest = TextIndexManifest.model_validate(read_json(manifest_path))
        chunks_path = stage_dir / manifest.chunks_path
        if (
            manifest.fingerprint == fingerprint
            and chunks_path.is_file()
            and len(read_jsonl(chunks_path)) == manifest.chunk_count
        ):
            return TextIndexResult(
                manifest_path=str(manifest_path), manifest=manifest, reused=True
            )

    materialized_pages = load_materialized_pages(source_manifest_path)
    pages = [entry.page for entry in materialized_pages]
    if text_source == "ocr":
        assert ocr_manifest_path is not None
        ocr_pages = load_ocr_pages(ocr_manifest_path)
        pages = [
            page.model_copy(update={"text": ocr_pages[page.page_id].text})
            for page in pages
            if page.page_id in ocr_pages
        ]
    chunks = _chunk_pages(
        pages,
        policy=policy,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise ValueError("index construction produced no chunks")

    stage_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = stage_dir / "chunks.jsonl"
    write_jsonl(chunks_path, (chunk.model_dump(mode="json") for chunk in chunks))
    manifest = TextIndexManifest(
        source_manifest_path=str(source_manifest_path),
        source_fingerprint=source_manifest.fingerprint,
        fingerprint=fingerprint,
        text_source=text_source,
        ocr_manifest_path=str(ocr_manifest_path) if ocr_manifest_path else None,
        chunk_policy=policy,
        chunk_size=chunk_size,
        overlap=overlap,
        page_count=len(pages),
        chunk_count=len(chunks),
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return TextIndexResult(manifest_path=str(manifest_path), manifest=manifest, reused=False)


def load_chunks(manifest_path: Path) -> list[ChunkRecord]:
    manifest = TextIndexManifest.model_validate(read_json(manifest_path))
    rows = read_jsonl(manifest_path.parent / manifest.chunks_path)
    chunks = [ChunkRecord.model_validate(row) for row in rows]
    if len(chunks) != manifest.chunk_count:
        raise ValueError("chunk artifact count does not match its manifest")
    return chunks


def _chunk_pages(
    pages: list[PageRecord],
    *,
    policy: ChunkPolicy,
    chunk_size: int,
    overlap: int,
) -> list[ChunkRecord]:
    if policy == "fixed":
        return FixedTokenChunker(chunk_size=chunk_size, overlap=overlap).chunk_pages(pages)
    return StructureAwareChunker(
        max_tokens=chunk_size, paragraph_overlap=overlap
    ).chunk_pages(pages)
