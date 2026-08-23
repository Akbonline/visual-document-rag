"""Evaluation contracts and failure attribution."""

from vidore_rag.evaluation.analysis import (
    EvidenceTypeBreakdown,
    EvidenceTypeMetrics,
    group_retrieval_by_evidence_type,
)
from vidore_rag.evaluation.benchmark import (
    BenchmarkReport,
    BenchmarkSession,
    QueryBenchmarkResult,
)
from vidore_rag.evaluation.metrics import (
    AggregateRetrievalMetrics,
    MetricStatus,
    QueryRetrievalMetrics,
    aggregate_metrics,
    judgments_by_query,
    score_query,
)

__all__ = [
    "AggregateRetrievalMetrics",
    "BenchmarkReport",
    "BenchmarkSession",
    "EvidenceTypeBreakdown",
    "EvidenceTypeMetrics",
    "MetricStatus",
    "QueryBenchmarkResult",
    "QueryRetrievalMetrics",
    "aggregate_metrics",
    "group_retrieval_by_evidence_type",
    "judgments_by_query",
    "score_query",
]
