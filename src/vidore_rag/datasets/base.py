from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from vidore_rag.contracts import EvaluationCapabilities, JudgmentRecord, QueryRecord
from vidore_rag.document_ir import PageRecord


class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> DatasetDescriptor:
        """Return immutable source identity needed by artifact manifests."""

    @abstractmethod
    def iter_page_assets(self) -> Iterable[PageAsset]:
        """Yield canonical pages together with their rendered source images."""

    @abstractmethod
    def iter_pages(self) -> Iterable[PageRecord]:
        """Yield canonical corpus pages."""

    @abstractmethod
    def capabilities(self) -> EvaluationCapabilities:
        """Return verified evaluation capabilities for the selected dataset revision."""

    @abstractmethod
    def iter_queries(self) -> Iterable[QueryRecord]:
        """Yield canonical queries."""

    @abstractmethod
    def iter_judgments(self) -> Iterable[JudgmentRecord]:
        """Yield canonical relevance judgments."""


class DatasetDescriptor(BaseModel):
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    language: str = Field(min_length=1)


@dataclass(frozen=True)
class PageAsset:
    source_key: str
    page: PageRecord
    image: Any

    def __post_init__(self) -> None:
        if not self.source_key or any(character in self.source_key for character in "/\\"):
            raise ValueError("page asset source_key must be a non-empty filename-safe value")
