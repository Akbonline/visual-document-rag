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


def classify_failure(
    *,
    gold_known: bool,
    is_answerable: bool | None,
    system_refused: bool,
    retrieved_coverage: float,
    context_coverage: float,
    answer_correct: bool,
    citation_correct: bool,
) -> FailureClass:
    if not gold_known or is_answerable is None:
        return FailureClass.UNSCORABLE_GOLD
    if not is_answerable:
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
