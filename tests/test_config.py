from pathlib import Path

from vidore_rag.config import inspect_dataset_config
from vidore_rag.datasets import build_default_registry
from vidore_rag.generation import load_generation_selection

ROOT = Path(__file__).resolve().parents[1]


def test_verified_vidore_config_is_ready() -> None:
    registry = build_default_registry()
    inspection = inspect_dataset_config(
        ROOT / "configs/datasets/vidore_v3.yaml",
        available_adapters=registry.names(),
    )

    assert inspection.ready
    assert inspection.config is not None
    assert inspection.config.dataset.dataset_id == "vidore/vidore_v3_hr"
    assert inspection.config.dataset.query_language == "english"
    assert inspection.config.evaluation.capabilities.has_reference_answers


def test_schema_valid_fixture_without_registered_adapter_is_not_ready() -> None:
    inspection = inspect_dataset_config(
        ROOT / "tests/fixtures/ready_dataset.yaml",
        available_adapters=build_default_registry().names(),
    )

    assert inspection.schema_valid
    assert not inspection.adapter_registered
    assert not inspection.ready
    assert inspection.config is not None
    assert inspection.config.dataset.dataset_id == "local/synthetic-docs"
    assert inspection.config.evaluation.capabilities.has_reference_answers


def test_gate_c_selection_is_deterministic_and_bound_to_the_dataset() -> None:
    selection = load_generation_selection(
        ROOT / "configs/experiments/gate_c.yaml"
    )

    assert selection.name == "gate-c-stratified-24"
    assert selection.deterministic_seed == 20260823
    assert len(selection.native_query_ids) == 24
    assert len(set(selection.native_query_ids)) == 24
    assert selection.dataset_fingerprint == (
        "c30f10d4d97b89e4d04d762683aabaa18de86731ef2bbbc7f1d75f87f4a2ebe3"
    )
