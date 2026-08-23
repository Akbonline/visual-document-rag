from pathlib import Path

from typer.testing import CliRunner

from vidore_rag.cli import app

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
