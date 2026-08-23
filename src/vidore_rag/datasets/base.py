from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from vidore_rag.contracts import EvaluationCapabilities, JudgmentRecord, QueryRecord
from vidore_rag.document_ir import PageRecord


class DatasetAdapter(ABC):
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
