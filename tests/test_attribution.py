from vidore_rag.evaluation.attribution import (
    AnswerabilityStatus,
    FailureClass,
    classify_failure,
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
