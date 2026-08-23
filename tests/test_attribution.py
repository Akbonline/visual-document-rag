from vidore_rag.contracts import JudgmentRecord, JudgmentStructure, JudgmentUnit
from vidore_rag.evaluation.attribution import (
    AnswerabilityStatus,
    FailureClass,
    classify_failure,
    evidence_coverage,
    resolve_answerability,
)


def test_partial_retrieval_is_not_misclassified_as_generation() -> None:
    result = classify_failure(
        gold_known=True,
        answerability=AnswerabilityStatus.LABELED_ANSWERABLE,
        system_refused=False,
        retrieved_coverage=0.5,
        context_coverage=0.5,
        answer_correct=False,
        citation_correct=False,
    )

    assert result is FailureClass.RETRIEVAL_INCOMPLETE


def test_correct_refusal_is_not_retrieval_failure() -> None:
    result = classify_failure(
        gold_known=True,
        answerability=AnswerabilityStatus.LABELED_UNANSWERABLE,
        system_refused=True,
        retrieved_coverage=0.0,
        context_coverage=0.0,
        answer_correct=False,
        citation_correct=False,
    )

    assert result is FailureClass.CORRECT_REFUSAL


def test_context_drop_is_attributed_before_generation() -> None:
    result = classify_failure(
        gold_known=True,
        answerability=AnswerabilityStatus.LABELED_ANSWERABLE,
        system_refused=False,
        retrieved_coverage=1.0,
        context_coverage=0.5,
        answer_correct=False,
        citation_correct=False,
    )

    assert result is FailureClass.CONTEXT_INCOMPLETE


def test_reference_answer_infers_answerability_without_a_dataset_label() -> None:
    status = resolve_answerability(
        is_answerable=None,
        has_answerability_label=False,
        has_reference_answer=True,
    )

    assert status is AnswerabilityStatus.INFERRED_ANSWERABLE
    result = classify_failure(
        gold_known=True,
        answerability=status,
        system_refused=False,
        retrieved_coverage=1.0,
        context_coverage=1.0,
        answer_correct=True,
        citation_correct=True,
    )
    assert result is FailureClass.SUCCESS


def test_missing_required_answerability_label_remains_unscorable() -> None:
    status = resolve_answerability(
        is_answerable=None,
        has_answerability_label=True,
        has_reference_answer=True,
    )

    assert status is AnswerabilityStatus.UNKNOWN


def test_unknown_multi_page_semantics_are_not_assumed_all_of() -> None:
    judgments = [
        JudgmentRecord(
            judgment_id="j1",
            query_id="q1",
            target_id="p1",
            judgment_unit=JudgmentUnit.PAGE,
            relevance=1,
        ),
        JudgmentRecord(
            judgment_id="j2",
            query_id="q1",
            target_id="p2",
            judgment_unit=JudgmentUnit.PAGE,
            relevance=1,
        ),
    ]

    coverage = evidence_coverage(
        observed_page_ids={"p1"},
        relevant_judgments=judgments,
        judgment_structure=JudgmentStructure.UNKNOWN,
        requirement_groups=[],
    )
    result = classify_failure(
        gold_known=True,
        answerability=AnswerabilityStatus.INFERRED_ANSWERABLE,
        system_refused=False,
        retrieved_coverage=coverage,
        context_coverage=coverage,
        answer_correct=True,
        citation_correct=True,
    )

    assert coverage is None
    assert result is FailureClass.UNSCORABLE_GOLD


def test_any_of_semantics_need_only_one_relevant_page() -> None:
    judgments = [
        JudgmentRecord(
            judgment_id=f"j{index}",
            query_id="q1",
            target_id=f"p{index}",
            judgment_unit=JudgmentUnit.PAGE,
            relevance=1,
        )
        for index in (1, 2)
    ]

    coverage = evidence_coverage(
        observed_page_ids={"p1"},
        relevant_judgments=judgments,
        judgment_structure=JudgmentStructure.ANY_OF,
        requirement_groups=[],
    )

    assert coverage == 1
