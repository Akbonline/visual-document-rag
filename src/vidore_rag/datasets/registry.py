from __future__ import annotations

from collections.abc import Callable

from vidore_rag.contracts import ProjectDatasetConfig
from vidore_rag.datasets.base import DatasetAdapter

AdapterFactory = Callable[[ProjectDatasetConfig], DatasetAdapter]


class DatasetRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory) -> None:
        if name in self._factories:
            raise ValueError(f"dataset adapter {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, config: ProjectDatasetConfig) -> DatasetAdapter:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(f"unknown dataset adapter {name!r}; registered: {known}") from exc
        return factory(config)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
