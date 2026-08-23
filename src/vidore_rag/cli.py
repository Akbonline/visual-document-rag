from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from vidore_rag.chunking import FixedTokenChunker
from vidore_rag.config import inspect_dataset_config
from vidore_rag.context import ContextBuilder
from vidore_rag.datasets import ViDoReV3Adapter
from vidore_rag.document_ir import PageRecord
from vidore_rag.evaluation import BenchmarkSession
from vidore_rag.ingestion import materialize_vidore
from vidore_rag.ingestion.fingerprints import stage_fingerprint
from vidore_rag.ocr import TesseractEngine, compare_ocr_to_source, run_cached_ocr
from vidore_rag.retrieval import (
    BM25Index,
    build_dense_index,
    build_text_index,
    project_chunks_to_pages,
)

app = typer.Typer(no_args_is_help=True, help="V3 Beta visual-document RAG tools")
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect and load dataset configurations")
artifact_app = typer.Typer(no_args_is_help=True, help="Inspect artifact identities")
demo_app = typer.Typer(no_args_is_help=True, help="Run local dataset-independent demos")
ocr_app = typer.Typer(no_args_is_help=True, help="Run and inspect cached OCR")
index_app = typer.Typer(no_args_is_help=True, help="Build sparse and dense indexes")
benchmark_app = typer.Typer(no_args_is_help=True, help="Query and evaluate real datasets")
app.add_typer(dataset_app, name="dataset")
app.add_typer(artifact_app, name="artifacts")
app.add_typer(demo_app, name="demo")
app.add_typer(ocr_app, name="ocr")
app.add_typer(index_app, name="index")
app.add_typer(benchmark_app, name="benchmark")

ARTIFACT_ROOT = Path(".artifacts/vidore_v3_hr")


@dataset_app.command("inspect")
def inspect_dataset(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    inspection = inspect_dataset_config(config)
    typer.echo(json.dumps(inspection.model_dump(mode="json"), indent=2, sort_keys=True))
    if not inspection.ready:
        raise typer.Exit(code=2)


@dataset_app.command("materialize")
def materialize_dataset(
    output: Annotated[Path, typer.Option()] = ARTIFACT_ROOT / "materialized",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Materialize the frozen ViDoRe V3 HR corpus and canonical records."""

    adapter = ViDoReV3Adapter.from_huggingface()
    result = materialize_vidore(adapter, output, limit=limit)
    _echo_model(result)


@artifact_app.command("fingerprint")
def fingerprint(
    stage: Annotated[str, typer.Option()],
    implementation_version: Annotated[str, typer.Option()] = "1",
    model_revision: Annotated[str | None, typer.Option()] = None,
) -> None:
    digest = stage_fingerprint(
        stage_name=stage,
        implementation_version=implementation_version,
        config={},
        model_revision=model_revision,
    )
    typer.echo(digest)


@demo_app.command("retrieve")
def demo_retrieve(
    pages: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    question: Annotated[str, typer.Option()],
    limit: Annotated[int, typer.Option(min=1, max=100)] = 5,
    chunk_size: Annotated[int, typer.Option(min=1)] = 80,
    overlap: Annotated[int, typer.Option(min=0)] = 16,
) -> None:
    """Run the deterministic text baseline over canonical page records."""

    raw_pages = json.loads(pages.read_text())
    if not isinstance(raw_pages, list):
        raise typer.BadParameter("pages JSON must contain a list")
    page_records = [PageRecord.model_validate(raw_page) for raw_page in raw_pages]
    chunks = FixedTokenChunker(chunk_size=chunk_size, overlap=overlap).chunk_pages(page_records)
    chunk_hits = BM25Index(chunks).search(question, limit=limit * 3)
    page_hits = project_chunks_to_pages(chunk_hits, limit=limit)
    typer.echo(
        json.dumps(
            [hit.model_dump(mode="json") for hit in page_hits], indent=2, sort_keys=True
        )
    )


@ocr_app.command("run")
def ocr_run(
    source_manifest: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = ARTIFACT_ROOT / "ocr",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    workers: Annotated[int, typer.Option(min=1, max=16)] = 4,
    language: Annotated[str, typer.Option()] = "eng",
    psm: Annotated[int, typer.Option(min=0, max=13)] = 3,
) -> None:
    """OCR materialized pages with resumable page-level caches."""

    source = source_manifest or _latest_manifest(ARTIFACT_ROOT / "materialized")
    result = run_cached_ocr(
        source,
        output,
        engine=TesseractEngine(language=language, page_segmentation_mode=psm),
        limit=limit,
        workers=workers,
    )
    _echo_model(result)


@ocr_app.command("compare")
def ocr_compare(
    source_manifest: Annotated[Path | None, typer.Option()] = None,
    ocr_manifest: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compare OCR tokens with the dataset's shipped markdown text."""

    source = source_manifest or _latest_manifest(ARTIFACT_ROOT / "materialized")
    ocr = ocr_manifest or _latest_manifest(ARTIFACT_ROOT / "ocr")
    _echo_model(compare_ocr_to_source(source, ocr))


@index_app.command("build")
def index_build(
    source_manifest: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = ARTIFACT_ROOT / "text_indexes",
    policy: Annotated[str, typer.Option()] = "fixed",
    text_source: Annotated[str, typer.Option()] = "markdown",
    ocr_manifest: Annotated[Path | None, typer.Option()] = None,
    chunk_size: Annotated[int, typer.Option(min=16)] = 180,
    overlap: Annotated[int, typer.Option(min=0)] = 24,
) -> None:
    """Build deterministic chunks for BM25 and dense retrieval."""

    if policy not in {"fixed", "structure"}:
        raise typer.BadParameter("policy must be fixed or structure")
    if text_source not in {"markdown", "ocr"}:
        raise typer.BadParameter("text-source must be markdown or ocr")
    source = source_manifest or _latest_manifest(ARTIFACT_ROOT / "materialized")
    resolved_ocr = ocr_manifest
    if text_source == "ocr" and resolved_ocr is None:
        resolved_ocr = _latest_manifest(ARTIFACT_ROOT / "ocr")
    result = build_text_index(
        source,
        output,
        policy=policy,  # type: ignore[arg-type]
        text_source=text_source,  # type: ignore[arg-type]
        ocr_manifest_path=resolved_ocr,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    _echo_model(result)


@index_app.command("dense")
def index_dense(
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = ARTIFACT_ROOT / "dense_indexes",
    model_name: Annotated[str, typer.Option()] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    ),
    model_revision: Annotated[str, typer.Option()] = (
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    ),
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Embed the text index into a persistent normalized dense matrix."""

    text_manifest = text_index_manifest or _latest_manifest(
        ARTIFACT_ROOT / "text_indexes"
    )
    result = build_dense_index(
        text_manifest,
        output,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        device=device,
    )
    _echo_model(result)


@benchmark_app.command("query")
def benchmark_query(
    query_id: Annotated[int, typer.Option(min=0)],
    mode: Annotated[str, typer.Option()] = "bm25",
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    dense_manifest: Annotated[Path | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    context_budget: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Run one shipped query and compare retrieved pages with gold pages."""

    session = _benchmark_session(mode, text_index_manifest, dense_manifest)
    result = session.query_by_native_id(query_id, limit=limit)
    payload = result.model_dump(mode="json")
    if context_budget is not None:
        payload["context"] = ContextBuilder(token_budget=context_budget).build(
            result.hits
        ).model_dump(mode="json")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@benchmark_app.command("evaluate")
def benchmark_evaluate(
    mode: Annotated[str, typer.Option()] = "bm25",
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    dense_manifest: Annotated[Path | None, typer.Option()] = None,
    limit_queries: Annotated[int | None, typer.Option(min=1)] = None,
    top_k: Annotated[int, typer.Option(min=10, max=100)] = 10,
    output: Annotated[Path | None, typer.Option()] = None,
    summary_only: Annotated[bool, typer.Option()] = False,
) -> None:
    """Evaluate a retriever over the selected English query slice."""

    session = _benchmark_session(mode, text_index_manifest, dense_manifest)
    report = session.evaluate(limit_queries=limit_queries, top_k=top_k)
    payload = report.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    displayed = payload["aggregate"] if summary_only else payload
    typer.echo(json.dumps(displayed, indent=2, sort_keys=True))


def _benchmark_session(
    mode: str,
    text_index_manifest: Path | None,
    dense_manifest: Path | None,
) -> BenchmarkSession:
    if mode not in {"bm25", "dense", "hybrid"}:
        raise typer.BadParameter("mode must be bm25, dense, or hybrid")
    text_manifest = text_index_manifest or _latest_manifest(
        ARTIFACT_ROOT / "text_indexes"
    )
    resolved_dense = dense_manifest
    if mode in {"dense", "hybrid"} and resolved_dense is None:
        resolved_dense = _latest_manifest(ARTIFACT_ROOT / "dense_indexes")
    return BenchmarkSession(
        text_manifest,
        mode=mode,  # type: ignore[arg-type]
        dense_manifest_path=resolved_dense,
    )


def _latest_manifest(root: Path) -> Path:
    manifests = sorted(root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise typer.BadParameter(f"no artifact manifest found under {root}")
    return manifests[-1]


def _echo_model(model: BaseModel) -> None:
    payload = model.model_dump(mode="json")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
