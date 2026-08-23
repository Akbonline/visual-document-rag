from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from vidore_rag.contracts import ProjectDatasetConfig


class DatasetInspection(BaseModel):
    config_path: str
    schema_valid: bool
    adapter_registered: bool
    ready: bool
    unresolved_fields: list[str] = Field(default_factory=list)
    config: ProjectDatasetConfig | None = None


def _find_placeholders(value: Any, path: str = "") -> list[str]:
    if isinstance(value, str) and value.startswith("__") and value.endswith("__"):
        return [path or "<root>"]
    if isinstance(value, dict):
        unresolved: list[str] = []
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            unresolved.extend(_find_placeholders(nested, nested_path))
        return unresolved
    if isinstance(value, list):
        unresolved = []
        for index, nested in enumerate(value):
            unresolved.extend(_find_placeholders(nested, f"{path}[{index}]"))
        return unresolved
    return []


def inspect_dataset_config(
    path: Path, *, available_adapters: Collection[str] = ()
) -> DatasetInspection:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dataset configuration must be a YAML mapping")

    unresolved = sorted(_find_placeholders(raw))
    if unresolved:
        return DatasetInspection(
            config_path=str(path),
            schema_valid=False,
            adapter_registered=False,
            ready=False,
            unresolved_fields=unresolved,
        )

    validated = ProjectDatasetConfig.model_validate(raw)
    registered = validated.dataset.adapter in available_adapters
    return DatasetInspection(
        config_path=str(path),
        schema_valid=True,
        adapter_registered=registered,
        ready=registered,
        config=validated,
    )
