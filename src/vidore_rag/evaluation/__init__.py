"""Evaluation contracts and failure attribution."""

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
    "MetricStatus",
    "QueryBenchmarkResult",
    "QueryRetrievalMetrics",
    "aggregate_metrics",
    "judgments_by_query",
    "score_query",
]
