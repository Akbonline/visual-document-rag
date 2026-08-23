from __future__ import annotations

from enum import StrEnum


class FailureClass(StrEnum):
    SUCCESS = "success"
    UNSCORABLE_GOLD = "unscorable_gold"
    CORRECT_REFUSAL = "correct_refusal"
    GENERATION_FALSE_ANSWER = "generation_false_answer"
    RETRIEVAL_INCOMPLETE = "retrieval_incomplete"
    CONTEXT_INCOMPLETE = "context_incomplete"
    GENERATION_FAILURE = "generation_failure"
    CITATION_FAILURE = "citation_failure"
    PERFORMANCE_FAILURE = "performance_failure"


class AnswerabilityStatus(StrEnum):
    LABELED_ANSWERABLE = "labeled_answerable"
    LABELED_UNANSWERABLE = "labeled_unanswerable"
    INFERRED_ANSWERABLE = "inferred_answerable"
    UNKNOWN = "unknown"


def resolve_answerability(
    *,
    is_answerable: bool | None,
    has_answerability_label: bool,
    has_reference_answer: bool,
) -> AnswerabilityStatus:
    if has_answerability_label:
        if is_answerable is True:
            return AnswerabilityStatus.LABELED_ANSWERABLE
        if is_answerable is False:
            return AnswerabilityStatus.LABELED_UNANSWERABLE
        return AnswerabilityStatus.UNKNOWN
    if has_reference_answer:
        return AnswerabilityStatus.INFERRED_ANSWERABLE
    return AnswerabilityStatus.UNKNOWN


def classify_failure(
    *,
    gold_known: bool,
    answerability: AnswerabilityStatus,
    system_refused: bool,
    retrieved_coverage: float,
    context_coverage: float,
    answer_correct: bool,
    citation_correct: bool,
) -> FailureClass:
    if not gold_known or answerability is AnswerabilityStatus.UNKNOWN:
        return FailureClass.UNSCORABLE_GOLD
    if answerability is AnswerabilityStatus.LABELED_UNANSWERABLE:
        return (
            FailureClass.CORRECT_REFUSAL
            if system_refused
            else FailureClass.GENERATION_FALSE_ANSWER
        )
    if retrieved_coverage < 1.0:
        return FailureClass.RETRIEVAL_INCOMPLETE
    if context_coverage < 1.0:
        return FailureClass.CONTEXT_INCOMPLETE
    if not answer_correct:
        return FailureClass.GENERATION_FAILURE
    if not citation_correct:
        return FailureClass.CITATION_FAILURE
    return FailureClass.SUCCESS
