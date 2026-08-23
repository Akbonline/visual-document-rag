from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field

from vidore_rag.artifacts import read_json, write_json
from vidore_rag.ingestion.fingerprints import stage_fingerprint
from vidore_rag.ingestion.materialize import (
    MaterializationManifest,
    MaterializedPage,
    load_materialized_pages,
)


class OCRWord(BaseModel):
    text: str = Field(min_length=1)
    confidence: float
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    block_number: int = Field(ge=0)
    paragraph_number: int = Field(ge=0)
    line_number: int = Field(ge=0)


class OCRPage(BaseModel):
    schema_version: str = "1"
    page_id: str
    image_sha256: str = Field(min_length=64, max_length=64)
    fingerprint: str = Field(min_length=64, max_length=64)
    engine: str
    engine_revision: str
    language: str
    page_segmentation_mode: int
    text: str
    words: list[OCRWord]
    duration_ms: float = Field(ge=0)


class OCRManifest(BaseModel):
    schema_version: str = "1"
    source_manifest_path: str
    source_fingerprint: str = Field(min_length=64, max_length=64)
    fingerprint: str = Field(min_length=64, max_length=64)
    engine: str
    engine_revision: str
    language: str
    page_segmentation_mode: int
    requested_page_count: int = Field(ge=0)
    completed_page_count: int = Field(ge=0)
    failed_pages: dict[str, str] = Field(default_factory=dict)
    pages_dir: str = "pages"


class OCRRunResult(BaseModel):
    manifest_path: str
    manifest: OCRManifest
    reused_page_count: int = Field(ge=0)
    processed_page_count: int = Field(ge=0)


class OCRComparison(BaseModel):
    page_count: int = Field(ge=0)
    source_token_count: int = Field(ge=0)
    ocr_token_count: int = Field(ge=0)
    token_precision: float = Field(ge=0, le=1)
    token_recall: float = Field(ge=0, le=1)
    token_f1: float = Field(ge=0, le=1)


class TesseractEngine:
    def __init__(
        self,
        *,
        executable: str = "tesseract",
        language: str = "eng",
        page_segmentation_mode: int = 3,
    ) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(f"Tesseract executable not found: {executable}")
        self.executable = resolved
        self.language = language
        self.page_segmentation_mode = page_segmentation_mode
        version = subprocess.run(
            [self.executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        self.revision = version.strip()

    def extract(self, image_path: Path, page: MaterializedPage) -> OCRPage:
        fingerprint = self.page_fingerprint(page)
        started = time.perf_counter()
        completed = subprocess.run(
            [
                self.executable,
                str(image_path),
                "stdout",
                "-l",
                self.language,
                "--psm",
                str(self.page_segmentation_mode),
                "tsv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        words, text = _parse_tsv(completed.stdout)
        return OCRPage(
            page_id=page.page.page_id,
            image_sha256=page.image_sha256,
            fingerprint=fingerprint,
            engine="tesseract",
            engine_revision=self.revision,
            language=self.language,
            page_segmentation_mode=self.page_segmentation_mode,
            text=text,
            words=words,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def page_fingerprint(self, page: MaterializedPage) -> str:
        return stage_fingerprint(
            stage_name="ocr_page",
            implementation_version="1",
            config={
                "language": self.language,
                "page_segmentation_mode": self.page_segmentation_mode,
            },
            upstream_fingerprints=[page.image_sha256],
            model_revision=self.revision,
        )


def run_cached_ocr(
    source_manifest_path: Path,
    output_root: Path,
    *,
    engine: TesseractEngine,
    limit: int | None = None,
    workers: int = 4,
) -> OCRRunResult:
    source_manifest = MaterializationManifest.model_validate(read_json(source_manifest_path))
    pages = load_materialized_pages(source_manifest_path)
    if limit is not None:
        pages = pages[:limit]
    stage_fingerprint_value = stage_fingerprint(
        stage_name="ocr",
        implementation_version="1",
        config={
            "language": engine.language,
            "page_segmentation_mode": engine.page_segmentation_mode,
            "limit": limit,
        },
        upstream_fingerprints=[source_manifest.fingerprint],
        model_revision=engine.revision,
    )
    stage_dir = output_root / stage_fingerprint_value
    page_dir = stage_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    reused = 0
    pending: list[MaterializedPage] = []
    for page in pages:
        output_path = page_dir / _page_filename(page)
        if output_path.exists():
            try:
                cached = OCRPage.model_validate(read_json(output_path))
            except (OSError, ValueError):
                pending.append(page)
                continue
            if cached.fingerprint == engine.page_fingerprint(page):
                reused += 1
                continue
        pending.append(page)

    failures: dict[str, str] = {}
    processed = 0
    maximum_workers = max(1, min(workers, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(
                engine.extract,
                source_manifest_path.parent / page.image_path,
                page,
            ): page
            for page in pending
        }
        for future in as_completed(futures):
            page = futures[future]
            try:
                result = future.result()
                write_json(page_dir / _page_filename(page), result.model_dump(mode="json"))
                processed += 1
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                failures[page.page.page_id] = str(exc)

    completed_count = reused + processed
    manifest = OCRManifest(
        source_manifest_path=str(source_manifest_path),
        source_fingerprint=source_manifest.fingerprint,
        fingerprint=stage_fingerprint_value,
        engine="tesseract",
        engine_revision=engine.revision,
        language=engine.language,
        page_segmentation_mode=engine.page_segmentation_mode,
        requested_page_count=len(pages),
        completed_page_count=completed_count,
        failed_pages=failures,
    )
    manifest_path = stage_dir / "manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return OCRRunResult(
        manifest_path=str(manifest_path),
        manifest=manifest,
        reused_page_count=reused,
        processed_page_count=processed,
    )


def load_ocr_pages(manifest_path: Path) -> dict[str, OCRPage]:
    manifest = OCRManifest.model_validate(read_json(manifest_path))
    page_dir = manifest_path.parent / manifest.pages_dir
    pages: dict[str, OCRPage] = {}
    for path in sorted(page_dir.glob("*.json")):
        page = OCRPage.model_validate(read_json(path))
        pages[page.page_id] = page
    return pages


def compare_ocr_to_source(
    source_manifest_path: Path, ocr_manifest_path: Path
) -> OCRComparison:
    materialized = {
        page.page.page_id: page for page in load_materialized_pages(source_manifest_path)
    }
    ocr_pages = load_ocr_pages(ocr_manifest_path)
    source_counts: Counter[str] = Counter()
    ocr_counts: Counter[str] = Counter()
    for page_id, ocr_page in ocr_pages.items():
        source_page = materialized.get(page_id)
        if source_page is None:
            continue
        source_counts.update(_comparison_tokens(source_page.page.text))
        ocr_counts.update(_comparison_tokens(ocr_page.text))

    overlap = sum((source_counts & ocr_counts).values())
    precision = overlap / sum(ocr_counts.values()) if ocr_counts else 0.0
    recall = overlap / sum(source_counts.values()) if source_counts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return OCRComparison(
        page_count=len(ocr_pages),
        source_token_count=sum(source_counts.values()),
        ocr_token_count=sum(ocr_counts.values()),
        token_precision=precision,
        token_recall=recall,
        token_f1=f1,
    )


def _parse_tsv(content: str) -> tuple[list[OCRWord], str]:
    words: list[OCRWord] = []
    lines: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for row in csv.DictReader(content.splitlines(), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        width = int(row["width"])
        height = int(row["height"])
        if width <= 0 or height <= 0:
            continue
        word = OCRWord(
            text=text,
            confidence=float(row["conf"]),
            x=int(row["left"]),
            y=int(row["top"]),
            width=width,
            height=height,
            block_number=int(row["block_num"]),
            paragraph_number=int(row["par_num"]),
            line_number=int(row["line_num"]),
        )
        words.append(word)
        lines[(word.block_number, word.paragraph_number, word.line_number)].append(word.text)
    text = "\n".join(" ".join(lines[key]) for key in sorted(lines))
    return words, text


def _comparison_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _page_filename(page: MaterializedPage) -> str:
    native_id = int(page.page.metadata["native_corpus_id"])
    return f"{native_id:06d}.json"
