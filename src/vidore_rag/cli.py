from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from vidore_rag.artifacts import read_json
from vidore_rag.chunking import FixedTokenChunker
from vidore_rag.config import inspect_dataset_config
from vidore_rag.context import ContextBuilder
from vidore_rag.datasets import build_default_registry
from vidore_rag.document_ir import PageRecord
from vidore_rag.evaluation import (
    BenchmarkSession,
    QueryRetrievalMetrics,
    group_retrieval_by_evidence_type,
)
from vidore_rag.generation import (
    GenerationExperimentManifest,
    GenerationRunner,
    build_provider,
    compare_generation_policies,
    load_generation_config,
    load_generation_results,
    load_generation_selection,
    run_generation_experiment,
    summarize_generation_by_evidence_type,
    summarize_generation_experiment,
)
from vidore_rag.ingestion import materialize_dataset as materialize_source
from vidore_rag.ingestion.fingerprints import stage_fingerprint
from vidore_rag.ingestion.materialize import load_judgments
from vidore_rag.ocr import TesseractEngine, compare_ocr_to_source, run_cached_ocr
from vidore_rag.retrieval import (
    BM25Index,
    DenseIndexManifest,
    TextIndexManifest,
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
generation_app = typer.Typer(
    no_args_is_help=True, help="Run retrieved and oracle answer experiments"
)
app.add_typer(dataset_app, name="dataset")
app.add_typer(artifact_app, name="artifacts")
app.add_typer(demo_app, name="demo")
app.add_typer(ocr_app, name="ocr")
app.add_typer(index_app, name="index")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(generation_app, name="generation")

ARTIFACT_ROOT = Path(".artifacts/vidore_v3_hr")
DEFAULT_DATASET_CONFIG = Path("configs/datasets/vidore_v3.yaml")
DEFAULT_GENERATION_CONFIG = Path("configs/generation/openai.yaml")
DATASET_REGISTRY = build_default_registry()


@dataset_app.command("inspect")
def inspect_dataset(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
) -> None:
    inspection = inspect_dataset_config(config, available_adapters=DATASET_REGISTRY.names())
    typer.echo(json.dumps(inspection.model_dump(mode="json"), indent=2, sort_keys=True))
    if not inspection.ready:
        raise typer.Exit(code=2)


@dataset_app.command("materialize")
def materialize_dataset(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)] = (
        DEFAULT_DATASET_CONFIG
    ),
    adapter: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = ARTIFACT_ROOT / "materialized",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Materialize any registered dataset through the shared adapter contract."""

    inspection = inspect_dataset_config(config, available_adapters=DATASET_REGISTRY.names())
    if not inspection.schema_valid or inspection.config is None:
        raise typer.BadParameter("dataset configuration is not schema-valid")
    adapter_name = adapter or inspection.config.dataset.adapter
    if adapter_name not in DATASET_REGISTRY.names():
        raise typer.BadParameter(
            f"adapter {adapter_name!r} is not registered; "
            f"available: {', '.join(DATASET_REGISTRY.names())}"
        )
    source = DATASET_REGISTRY.create(adapter_name, inspection.config)
    result = materialize_source(source, output, limit=limit)
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
        json.dumps([hit.model_dump(mode="json") for hit in page_hits], indent=2, sort_keys=True)
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
    model_name: Annotated[str, typer.Option()] = ("sentence-transformers/all-MiniLM-L6-v2"),
    model_revision: Annotated[str, typer.Option()] = ("1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Embed the text index into a persistent normalized dense matrix."""

    text_manifest = text_index_manifest or _latest_manifest(ARTIFACT_ROOT / "text_indexes")
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
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    mode: Annotated[str, typer.Option()] = "bm25",
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    dense_manifest: Annotated[Path | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    context_budget: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Run one shipped query and compare retrieved pages with gold pages."""

    session = _benchmark_session(mode, text_index_manifest, dense_manifest, dataset_config)
    result = session.query_by_native_id(query_id, limit=limit)
    payload = result.model_dump(mode="json")
    if context_budget is not None:
        payload["context"] = (
            ContextBuilder(token_budget=context_budget).build(result.hits).model_dump(mode="json")
        )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@generation_app.command("compare")
def generation_compare(
    baseline_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    candidate_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compare two generation runs over the same query and model contract."""

    baseline_contract = GenerationExperimentManifest.model_validate(read_json(baseline_manifest))
    candidate_contract = GenerationExperimentManifest.model_validate(read_json(candidate_manifest))
    if baseline_contract.dataset_fingerprint != candidate_contract.dataset_fingerprint:
        raise typer.BadParameter("generation manifests use different datasets")
    if baseline_contract.native_query_ids != candidate_contract.native_query_ids:
        raise typer.BadParameter("generation manifests use different query selections")
    if baseline_contract.config != candidate_contract.config:
        raise typer.BadParameter("generation manifests use different generation configs")
    comparison = compare_generation_policies(
        load_generation_results(baseline_manifest),
        load_generation_results(candidate_manifest),
    )
    payload = comparison.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@benchmark_app.command("evaluate")
def benchmark_evaluate(
    mode: Annotated[str, typer.Option()] = "bm25",
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    dense_manifest: Annotated[Path | None, typer.Option()] = None,
    limit_queries: Annotated[int | None, typer.Option(min=1)] = None,
    top_k: Annotated[int, typer.Option(min=10, max=100)] = 10,
    output: Annotated[Path | None, typer.Option()] = None,
    summary_only: Annotated[bool, typer.Option()] = False,
) -> None:
    """Evaluate a retriever over the selected English query slice."""

    session = _benchmark_session(mode, text_index_manifest, dense_manifest, dataset_config)
    report = session.evaluate(limit_queries=limit_queries, top_k=top_k)
    payload = report.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    displayed = payload["aggregate"] if summary_only else payload
    typer.echo(json.dumps(displayed, indent=2, sort_keys=True))


@benchmark_app.command("evidence-breakdown")
def benchmark_evidence_breakdown(
    results: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    source_manifest: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = None,
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Group per-query retrieval quality by gold evidence content type."""

    inspection = inspect_dataset_config(dataset_config, available_adapters=DATASET_REGISTRY.names())
    if not inspection.schema_valid or inspection.config is None:
        raise typer.BadParameter("dataset configuration is not schema-valid")
    threshold = inspection.config.evaluation.capabilities.binary_relevance_threshold
    if threshold is None:
        raise typer.BadParameter("dataset has no binary relevance threshold")
    resolved_source = source_manifest or _latest_manifest(ARTIFACT_ROOT / "materialized")
    raw_results = read_json(results)
    query_metrics = [QueryRetrievalMetrics.model_validate(row) for row in raw_results["queries"]]
    breakdown = group_retrieval_by_evidence_type(
        query_metrics,
        load_judgments(resolved_source),
        binary_relevance_threshold=threshold,
    )
    payload = breakdown.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@generation_app.command("query")
def generation_query(
    query_id: Annotated[int, typer.Option(min=0)],
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_GENERATION_CONFIG,
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    mode: Annotated[str, typer.Option()] = "hybrid",
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    dense_manifest: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate paired retrieved-context and oracle-context answers."""

    generation_config = load_generation_config(config)
    session = _benchmark_session(mode, text_index_manifest, dense_manifest, dataset_config)
    try:
        provider = build_provider(generation_config.provider)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    result = GenerationRunner(session, provider, generation_config).run_pair(query_id)
    _echo_model(result)


@generation_app.command("run")
def generation_run(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_GENERATION_CONFIG,
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    mode: Annotated[str, typer.Option()] = "hybrid",
    text_index_manifest: Annotated[Path | None, typer.Option()] = None,
    dense_manifest: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = ARTIFACT_ROOT / "generation",
    limit_queries: Annotated[int | None, typer.Option(min=1)] = None,
    selection: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = None,
) -> None:
    """Run a resumable paired generation experiment over the selected queries."""

    generation_config = load_generation_config(config)
    session = _benchmark_session(mode, text_index_manifest, dense_manifest, dataset_config)
    try:
        provider = build_provider(generation_config.provider)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if limit_queries is not None and selection is not None:
        raise typer.BadParameter("--limit-queries and --selection are mutually exclusive")
    available_ids = {int(query.metadata["native_query_id"]) for query in session.queries}
    query_ids = sorted(available_ids)
    if selection is not None:
        selected = load_generation_selection(selection)
        if selected.dataset_fingerprint != session.text_manifest.source_fingerprint:
            raise typer.BadParameter("selection dataset fingerprint does not match the index")
        missing = set(selected.native_query_ids) - available_ids
        if missing:
            raise typer.BadParameter(
                f"selection contains query IDs outside this dataset slice: {sorted(missing)}"
            )
        query_ids = selected.native_query_ids
    elif limit_queries is not None:
        query_ids = query_ids[:limit_queries]
    runner = GenerationRunner(session, provider, generation_config)
    manifest = run_generation_experiment(
        runner,
        provider,
        generation_config,
        output,
        native_query_ids=query_ids,
        on_progress=lambda completed, total, message: typer.echo(
            f"[{completed}/{total}] {message}", err=True
        ),
    )
    payload = manifest.model_dump(mode="json")
    payload["manifest_path"] = str(output / manifest.fingerprint / "manifest.json")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@generation_app.command("summarize")
def generation_summarize(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    source_manifest: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = None,
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Summarize completed generation pairs under declared gold semantics."""

    inspection = inspect_dataset_config(dataset_config, available_adapters=DATASET_REGISTRY.names())
    if not inspection.schema_valid or inspection.config is None:
        raise typer.BadParameter("dataset configuration is not schema-valid")
    resolved_source = source_manifest or _latest_manifest(ARTIFACT_ROOT / "materialized")
    summary = summarize_generation_experiment(
        load_generation_results(manifest),
        load_judgments(resolved_source),
        inspection.config.evaluation.capabilities,
    )
    payload = summary.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@generation_app.command("evidence-breakdown")
def generation_evidence_breakdown(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    source_manifest: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = None,
    dataset_config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, readable=True)
    ] = DEFAULT_DATASET_CONFIG,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Group generation quality by mutually exclusive gold evidence stratum."""

    inspection = inspect_dataset_config(dataset_config, available_adapters=DATASET_REGISTRY.names())
    if not inspection.schema_valid or inspection.config is None:
        raise typer.BadParameter("dataset configuration is not schema-valid")
    resolved_source = source_manifest or _latest_manifest(ARTIFACT_ROOT / "materialized")
    summary = summarize_generation_by_evidence_type(
        load_generation_results(manifest),
        load_judgments(resolved_source),
        inspection.config.evaluation.capabilities,
    )
    payload = summary.model_dump(mode="json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _benchmark_session(
    mode: str,
    text_index_manifest: Path | None,
    dense_manifest: Path | None,
    dataset_config: Path,
) -> BenchmarkSession:
    if mode not in {"bm25", "dense", "hybrid"}:
        raise typer.BadParameter("mode must be bm25, dense, or hybrid")
    text_manifest, resolved_dense = _resolve_index_manifests(
        mode, text_index_manifest, dense_manifest
    )
    inspection = inspect_dataset_config(dataset_config, available_adapters=DATASET_REGISTRY.names())
    if not inspection.schema_valid or inspection.config is None:
        raise typer.BadParameter("dataset configuration is not schema-valid")
    return BenchmarkSession(
        text_manifest,
        mode=mode,  # type: ignore[arg-type]
        dense_manifest_path=resolved_dense,
        capabilities=inspection.config.evaluation.capabilities,
    )


def _resolve_index_manifests(
    mode: str,
    text_index_manifest: Path | None,
    dense_manifest: Path | None,
) -> tuple[Path, Path | None]:
    text_root = ARTIFACT_ROOT / "text_indexes"
    dense_root = ARTIFACT_ROOT / "dense_indexes"
    if mode == "bm25":
        return text_index_manifest or _latest_manifest(text_root), None

    resolved_dense = dense_manifest
    resolved_text = text_index_manifest
    if resolved_dense is None and resolved_text is None:
        resolved_dense = _latest_manifest(dense_root)

    if resolved_dense is not None:
        dense_contract = DenseIndexManifest.model_validate(read_json(resolved_dense))
        if resolved_text is None:
            recorded_text_path = Path(dense_contract.text_index_manifest_path)
            resolved_text = (
                recorded_text_path
                if recorded_text_path.is_file()
                else _find_text_manifest(dense_contract.text_index_fingerprint, text_root)
            )
    else:
        assert resolved_text is not None
        text_contract = TextIndexManifest.model_validate(read_json(resolved_text))
        resolved_dense = _find_dense_manifest(text_contract.fingerprint, dense_root)
        dense_contract = DenseIndexManifest.model_validate(read_json(resolved_dense))

    assert resolved_text is not None and resolved_dense is not None
    text_contract = TextIndexManifest.model_validate(read_json(resolved_text))
    if dense_contract.text_index_fingerprint != text_contract.fingerprint:
        raise typer.BadParameter(
            "dense and text index fingerprints do not match; pass a compatible "
            "--text-index-manifest and --dense-manifest pair"
        )
    return resolved_text, resolved_dense


def _find_text_manifest(fingerprint: str, root: Path) -> Path:
    for path in _manifests_newest_first(root):
        manifest = TextIndexManifest.model_validate(read_json(path))
        if manifest.fingerprint == fingerprint:
            return path
    raise typer.BadParameter(
        f"no text index matches dense index fingerprint {fingerprint} under {root}"
    )


def _find_dense_manifest(text_fingerprint: str, root: Path) -> Path:
    for path in _manifests_newest_first(root):
        manifest = DenseIndexManifest.model_validate(read_json(path))
        if manifest.text_index_fingerprint == text_fingerprint:
            return path
    raise typer.BadParameter(
        f"no dense index matches text index fingerprint {text_fingerprint} under {root}"
    )


def _manifests_newest_first(root: Path) -> list[Path]:
    return sorted(
        root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _latest_manifest(root: Path) -> Path:
    manifests = _manifests_newest_first(root)
    if not manifests:
        raise typer.BadParameter(f"no artifact manifest found under {root}")
    return manifests[0]


def _echo_model(model: BaseModel) -> None:
    payload = model.model_dump(mode="json")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
