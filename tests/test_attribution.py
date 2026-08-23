from vidore_rag.evaluation.attribution import FailureClass, classify_failure


def test_partial_retrieval_is_not_misclassified_as_generation() -> None:
    result = classify_failure(
        gold_known=True,
        is_answerable=True,
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
        is_answerable=False,
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
        is_answerable=True,
        system_refused=False,
        retrieved_coverage=1.0,
        context_coverage=0.5,
        answer_correct=False,
        citation_correct=False,
    )

    assert result is FailureClass.CONTEXT_INCOMPLETE

