from pathlib import Path

import pytest
from typer.testing import CliRunner

import vidore_rag.cli as cli
from vidore_rag.artifacts import write_json
from vidore_rag.cli import app
from vidore_rag.retrieval import DenseIndexManifest, TextIndexManifest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def test_cli_registers_all_public_command_groups() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "dataset",
        "artifacts",
        "demo",
        "ocr",
        "index",
        "benchmark",
        "generation",
    ):
        assert command in result.stdout


def test_dataset_inspect_distinguishes_validity_from_loadability() -> None:
    verified = RUNNER.invoke(
        app,
        [
            "dataset",
            "inspect",
            "--config",
            str(ROOT / "configs/datasets/vidore_v3.yaml"),
        ],
    )
    fixture = RUNNER.invoke(
        app,
        [
            "dataset",
            "inspect",
            "--config",
            str(ROOT / "tests/fixtures/ready_dataset.yaml"),
        ],
    )

    assert verified.exit_code == 0
    assert '"schema_valid": true' in verified.stdout
    assert '"adapter_registered": true' in verified.stdout
    assert fixture.exit_code == 2
    assert '"schema_valid": true' in fixture.stdout
    assert '"adapter_registered": false' in fixture.stdout


def test_documented_demo_returns_expected_page() -> None:
    result = RUNNER.invoke(
        app,
        [
            "demo",
            "retrieve",
            "--pages",
            str(ROOT / "tests/fixtures/pages.json"),
            "--question",
            "What does ERR-42 mean?",
            "--chunk-size",
            "8",
            "--overlap",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert '"page_id": "manual:2"' in result.stdout


def test_hybrid_default_selects_the_text_index_recorded_by_dense_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    text_root = artifact_root / "text_indexes"
    dense_root = artifact_root / "dense_indexes"
    compatible_path = text_root / ("a" * 64) / "manifest.json"
    unrelated_path = text_root / ("b" * 64) / "manifest.json"
    dense_path = dense_root / ("c" * 64) / "manifest.json"
    for path in (compatible_path, unrelated_path, dense_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, fingerprint in (
        (compatible_path, "a" * 64),
        (unrelated_path, "b" * 64),
    ):
        manifest = TextIndexManifest(
            source_manifest_path="source.json",
            source_fingerprint="d" * 64,
            fingerprint=fingerprint,
            text_source="markdown",
            chunk_policy="fixed",
            chunk_size=180,
            overlap=24,
            page_count=1,
            chunk_count=1,
        )
        write_json(path, manifest.model_dump(mode="json"))
    dense = DenseIndexManifest(
        text_index_manifest_path=str(compatible_path),
        text_index_fingerprint="a" * 64,
        fingerprint="c" * 64,
        model_name="model",
        model_revision="revision",
        device="cpu",
        batch_size=1,
        chunk_count=1,
        dimensions=1,
    )
    write_json(dense_path, dense.model_dump(mode="json"))
    monkeypatch.setattr(cli, "ARTIFACT_ROOT", artifact_root)

    selected_text, selected_dense = cli._resolve_index_manifests(
        "hybrid", None, None
    )

    assert selected_text == compatible_path
    assert selected_dense == dense_path
