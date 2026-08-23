import pytest

from vidore_rag.contracts import (
    AnswerFormat,
    EvaluationCapabilities,
    JudgmentCardinality,
    JudgmentScale,
    JudgmentStructure,
    JudgmentUnit,
)
from vidore_rag.evaluation.metrics import MetricStatus, aggregate_metrics, score_query


def capabilities() -> EvaluationCapabilities:
    return EvaluationCapabilities(
        judgment_unit=JudgmentUnit.PAGE,
        judgment_cardinality=JudgmentCardinality.VARIABLE,
        judgment_scale=JudgmentScale.GRADED,
        relevance_grades=[0, 1, 2],
        binary_relevance_threshold=1,
        judgment_structure=JudgmentStructure.UNKNOWN,
        has_reference_answers=True,
        answer_format=AnswerFormat.FREE_TEXT,
        has_answerability_label=False,
        has_bounding_box_evidence=True,
    )


def test_retrieval_metrics_use_graded_gold_and_page_recall() -> None:
    metrics = score_query(
        query_id="q1",
        ranked_page_ids=["p2", "noise", "p1"],
        relevance={"p1": 2, "p2": 1},
        capabilities=capabilities(),
    )

    assert metrics.recall_at_1 == 0.5
    assert metrics.recall_at_5 == 1
    assert metrics.reciprocal_rank == 1
    assert 0 < metrics.ndcg_at_10 < 1


def test_metric_aggregation_reports_latency_percentiles() -> None:
    first = score_query(
        query_id="q1",
        ranked_page_ids=["p1"],
        relevance={"p1": 1},
        capabilities=capabilities(),
        latency_ms=10,
    )
    second = score_query(
        query_id="q2",
        ranked_page_ids=["p2"],
        relevance={"p2": 1},
        capabilities=capabilities(),
        latency_ms=30,
    )

    aggregate = aggregate_metrics([first, second])

    assert aggregate.mrr == 1
    assert aggregate.latency_p50_ms == pytest.approx(20)
    assert aggregate.latency_p95_ms == pytest.approx(29)


def test_empty_gold_is_not_applicable_instead_of_crashing() -> None:
    metrics = score_query(
        query_id="q-empty",
        ranked_page_ids=["p1"],
        relevance={},
        capabilities=capabilities(),
        latency_ms=4,
    )

    assert metrics.status is MetricStatus.NOT_APPLICABLE
    assert metrics.reason == "query has no gold judgments"
    aggregate = aggregate_metrics([metrics])
    assert aggregate.scored_query_count == 0
    assert aggregate.not_applicable_query_count == 1
    assert aggregate.ndcg_at_10 is None
    assert aggregate.latency_p50_ms == 4


def test_declared_relevance_threshold_controls_recall() -> None:
    strict = capabilities().model_copy(update={"binary_relevance_threshold": 2})
    metrics = score_query(
        query_id="q-threshold",
        ranked_page_ids=["p1", "p2"],
        relevance={"p1": 1, "p2": 2},
        capabilities=strict,
    )

    assert metrics.recall_at_1 == 0
    assert metrics.recall_at_5 == 1
