from __future__ import annotations

import math
import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

from vidore_rag.contracts import JudgmentRecord


class QueryRetrievalMetrics(BaseModel):
    query_id: str
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)


class AggregateRetrievalMetrics(BaseModel):
    query_count: int = Field(ge=1)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)


def score_query(
    *,
    query_id: str,
    ranked_page_ids: list[str],
    relevance: dict[str, float],
    latency_ms: float = 0,
    binary_threshold: float = 1,
) -> QueryRetrievalMetrics:
    relevant = {
        page_id for page_id, score in relevance.items() if score >= binary_threshold
    }
    if not relevant:
        raise ValueError(f"query {query_id} has no relevant pages at threshold")
    return QueryRetrievalMetrics(
        query_id=query_id,
        recall_at_1=_recall_at_k(ranked_page_ids, relevant, 1),
        recall_at_5=_recall_at_k(ranked_page_ids, relevant, 5),
        recall_at_10=_recall_at_k(ranked_page_ids, relevant, 10),
        reciprocal_rank=_reciprocal_rank(ranked_page_ids, relevant),
        ndcg_at_10=_ndcg_at_k(ranked_page_ids, relevance, 10),
        latency_ms=latency_ms,
    )


def aggregate_metrics(rows: list[QueryRetrievalMetrics]) -> AggregateRetrievalMetrics:
    if not rows:
        raise ValueError("cannot aggregate an empty metric set")
    latencies = sorted(row.latency_ms for row in rows)
    return AggregateRetrievalMetrics(
        query_count=len(rows),
        recall_at_1=statistics.fmean(row.recall_at_1 for row in rows),
        recall_at_5=statistics.fmean(row.recall_at_5 for row in rows),
        recall_at_10=statistics.fmean(row.recall_at_10 for row in rows),
        mrr=statistics.fmean(row.reciprocal_rank for row in rows),
        ndcg_at_10=statistics.fmean(row.ndcg_at_10 for row in rows),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )


def judgments_by_query(
    judgments: list[JudgmentRecord],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for judgment in judgments:
        grouped[judgment.query_id][judgment.target_id] = judgment.relevance
    return dict(grouped)


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, page_id in enumerate(ranked, start=1):
        if page_id in relevant:
            return 1 / rank
    return 0.0


def _ndcg_at_k(ranked: list[str], relevance: dict[str, float], k: int) -> float:
    actual = [relevance.get(page_id, 0.0) for page_id in ranked[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_score = _dcg(ideal)
    return _dcg(actual) / ideal_score if ideal_score else 0.0


def _dcg(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, 1))


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
