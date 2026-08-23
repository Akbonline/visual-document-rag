from pathlib import Path

from vidore_rag.config import inspect_dataset_config

ROOT = Path(__file__).resolve().parents[1]


def test_verified_vidore_config_is_ready() -> None:
    inspection = inspect_dataset_config(ROOT / "configs/datasets/vidore_v3.yaml")

    assert inspection.ready
    assert inspection.config is not None
    assert inspection.config.dataset.dataset_id == "vidore/vidore_v3_hr"
    assert inspection.config.dataset.query_language == "english"
    assert inspection.config.evaluation.capabilities.has_reference_answers


def test_ready_fixture_is_validated() -> None:
    inspection = inspect_dataset_config(ROOT / "tests/fixtures/ready_dataset.yaml")

    assert inspection.ready
    assert inspection.config is not None
    assert inspection.config.dataset.dataset_id == "local/synthetic-docs"
    assert inspection.config.evaluation.capabilities.has_reference_answers
