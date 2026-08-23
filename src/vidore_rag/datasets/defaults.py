from __future__ import annotations

from vidore_rag.contracts import ProjectDatasetConfig
from vidore_rag.datasets.registry import DatasetRegistry
from vidore_rag.datasets.vidore_v3 import ViDoReV3Adapter


def build_default_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register("vidore_v3", _create_vidore_v3)
    return registry


def _create_vidore_v3(config: ProjectDatasetConfig) -> ViDoReV3Adapter:
    selection = config.dataset
    expected = (
        ViDoReV3Adapter.DATASET_ID,
        ViDoReV3Adapter.REVISION,
        ViDoReV3Adapter.SPLIT,
    )
    actual = (selection.dataset_id, selection.revision, selection.split)
    if actual != expected:
        raise ValueError(
            "vidore_v3 configuration does not match the adapter's frozen source identity"
        )
    return ViDoReV3Adapter.from_huggingface(
        language=selection.query_language or "english"
    )
