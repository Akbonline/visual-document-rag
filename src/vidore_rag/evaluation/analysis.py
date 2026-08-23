from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from vidore_rag.contracts import JudgmentRecord
from vidore_rag.evaluation.metrics import MetricStatus, QueryRetrievalMetrics


class EvidenceTypeMetrics(BaseModel):
    evidence_type: str
    n_queries: int = Field(ge=0)
    ndcg_at_10: float | None = Field(default=None, ge=0, le=1)
    recall_at_10: float | None = Field(default=None, ge=0, le=1)


class EvidenceTypeBreakdown(BaseModel):
    query_membership: str = "multi_label"
    rows: list[EvidenceTypeMetrics]


def group_retrieval_by_evidence_type(
    query_metrics: list[QueryRetrievalMetrics],
    judgments: list[JudgmentRecord],
    *,
    binary_relevance_threshold: float,
    evidence_types: tuple[str, ...] = ("Text", "Table", "Chart", "Infographic"),
) -> EvidenceTypeBreakdown:
    query_types: dict[str, set[str]] = defaultdict(set)
    for judgment in judgments:
        if judgment.relevance < binary_relevance_threshold:
            continue
        query_types[judgment.query_id].update(
            _canonical_type(value) for value in judgment.content_types
        )
    metrics_by_query = {
        row.query_id: row
        for row in query_metrics
        if row.status is MetricStatus.APPLICABLE
    }
    rows: list[EvidenceTypeMetrics] = []
    for evidence_type in evidence_types:
        metrics = [
            metrics_by_query[query_id]
            for query_id, types in query_types.items()
            if evidence_type in types and query_id in metrics_by_query
        ]
        rows.append(
            EvidenceTypeMetrics(
                evidence_type=evidence_type,
                n_queries=len(metrics),
                ndcg_at_10=_mean([row.ndcg_at_10 for row in metrics]),
                recall_at_10=_mean([row.recall_at_10 for row in metrics]),
            )
        )
    return EvidenceTypeBreakdown(rows=rows)


def _canonical_type(value: str) -> str:
    return value.strip().title()


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
