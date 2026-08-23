import pytest

from vidore_rag.evaluation.metrics import aggregate_metrics, score_query


def test_retrieval_metrics_use_graded_gold_and_page_recall() -> None:
    metrics = score_query(
        query_id="q1",
        ranked_page_ids=["p2", "noise", "p1"],
        relevance={"p1": 2, "p2": 1},
    )

    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_5 == 1
    assert metrics.reciprocal_rank == 1
    assert 0 < metrics.ndcg_at_10 < 1


def test_metric_aggregation_reports_latency_percentiles() -> None:
    first = score_query(
        query_id="q1", ranked_page_ids=["p1"], relevance={"p1": 1}, latency_ms=10
    )
    second = score_query(
        query_id="q2", ranked_page_ids=["p2"], relevance={"p2": 1}, latency_ms=30
    )

    aggregate = aggregate_metrics([first, second])

    assert aggregate.mrr == 1
    assert aggregate.latency_p50_ms == pytest.approx(20)
    assert aggregate.latency_p95_ms == pytest.approx(29)
