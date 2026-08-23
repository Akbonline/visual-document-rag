from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vidore_rag.artifacts import read_json, write_json
from vidore_rag.generation.models import (
    GenerationConfig,
    GenerationExperimentManifest,
    GenerationJobFailure,
    GenerationPairResult,
)
from vidore_rag.generation.provider import LLMProvider, RateLimitedProvider
from vidore_rag.generation.runner import GenerationRunner
from vidore_rag.ingestion.fingerprints import stage_fingerprint


def run_generation_experiment(
    runner: GenerationRunner,
    provider: LLMProvider,
    config: GenerationConfig,
    output_root: Path,
    *,
    native_query_ids: list[int],
) -> GenerationExperimentManifest:
    selected_ids = sorted(set(native_query_ids))
    if not selected_ids:
        raise ValueError("generation experiment requires at least one query")
    session = runner.session
    fingerprint = stage_fingerprint(
        stage_name="generation_experiment",
        implementation_version="1",
        config={
            "generation": config.model_dump(mode="json"),
            "native_query_ids": selected_ids,
        },
        upstream_fingerprints=(
            session.text_manifest.source_fingerprint,
            session.text_manifest.fingerprint,
            session.projection.projection_fingerprint,
        ),
        model_revision=config.provider.model,
    )
    stage_dir = output_root / fingerprint
    results_dir = stage_dir / "queries"
    results_dir.mkdir(parents=True, exist_ok=True)
    completed = [query_id for query_id in selected_ids if _valid_result(results_dir, query_id)]
    pending = [query_id for query_id in selected_ids if query_id not in completed]
    failures: list[GenerationJobFailure] = []
    if pending:
        controlled_provider = RateLimitedProvider(
            provider, requests_per_second=config.requests_per_second
        )
        controlled_runner = GenerationRunner(session, controlled_provider, config)
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {
                executor.submit(controlled_runner.run_pair, query_id): query_id
                for query_id in pending
            }
            for future in as_completed(futures):
                query_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # provider failures remain pending for resume
                    failures.append(
                        GenerationJobFailure(
                            native_query_id=query_id,
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
                    )
                    continue
                write_json(
                    _result_path(results_dir, query_id), result.model_dump(mode="json")
                )
                completed.append(query_id)
    manifest = GenerationExperimentManifest(
        fingerprint=fingerprint,
        dataset_fingerprint=session.text_manifest.source_fingerprint,
        text_index_fingerprint=session.text_manifest.fingerprint,
        dense_index_fingerprint=(
            session.dense_manifest.fingerprint if session.dense_manifest else None
        ),
        projection_fingerprint=session.projection.projection_fingerprint,
        config=config,
        native_query_ids=selected_ids,
        completed_query_ids=sorted(completed),
        failures=sorted(failures, key=lambda row: row.native_query_id),
        results_directory="queries",
    )
    write_json(
        stage_dir / "manifest.json", manifest.model_dump(mode="json")
    )
    return manifest


def load_generation_results(manifest_path: Path) -> list[GenerationPairResult]:
    manifest = GenerationExperimentManifest.model_validate(read_json(manifest_path))
    results_dir = manifest_path.parent / manifest.results_directory
    return [
        GenerationPairResult.model_validate(read_json(_result_path(results_dir, query_id)))
        for query_id in manifest.completed_query_ids
    ]


def _result_path(results_dir: Path, native_query_id: int) -> Path:
    return results_dir / f"{native_query_id}.json"


def _valid_result(results_dir: Path, native_query_id: int) -> bool:
    path = _result_path(results_dir, native_query_id)
    if not path.is_file():
        return False
    try:
        result = GenerationPairResult.model_validate(read_json(path))
    except (OSError, ValueError):
        return False
    return result.native_query_id == native_query_id
