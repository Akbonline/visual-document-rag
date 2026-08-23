from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from vidore_rag.chunking import FixedTokenChunker
from vidore_rag.config import inspect_dataset_config
from vidore_rag.document_ir import PageRecord
from vidore_rag.ingestion.fingerprints import stage_fingerprint
from vidore_rag.retrieval import BM25Index, project_chunks_to_pages

app = typer.Typer(no_args_is_help=True, help="V3 Beta visual-document RAG tools")
dataset_app = typer.Typer(no_args_is_help=True, help="Inspect and load dataset configurations")
artifact_app = typer.Typer(no_args_is_help=True, help="Inspect artifact identities")
demo_app = typer.Typer(no_args_is_help=True, help="Run local dataset-independent demos")
app.add_typer(dataset_app, name="dataset")
app.add_typer(artifact_app, name="artifacts")
app.add_typer(demo_app, name="demo")


@dataset_app.command("inspect")
def inspect_dataset(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    inspection = inspect_dataset_config(config)
    typer.echo(json.dumps(inspection.model_dump(mode="json"), indent=2, sort_keys=True))
    if not inspection.ready:
        raise typer.Exit(code=2)


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
