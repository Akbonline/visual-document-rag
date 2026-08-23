from __future__ import annotations

from enum import StrEnum

from vidore_rag.contracts import EvidenceRequirementGroup, JudgmentRecord, JudgmentStructure


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
    retrieved_coverage: float | None,
    context_coverage: float | None,
    answer_correct: bool,
    citation_correct: bool,
) -> FailureClass:
    if (
        not gold_known
        or answerability is AnswerabilityStatus.UNKNOWN
        or retrieved_coverage is None
        or context_coverage is None
    ):
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


def evidence_coverage(
    *,
    observed_page_ids: set[str],
    relevant_judgments: list[JudgmentRecord],
    judgment_structure: JudgmentStructure,
    requirement_groups: list[EvidenceRequirementGroup],
) -> float | None:
    """Return completeness only when the gold evidence semantics make it knowable."""

    gold_page_ids = {judgment.target_id for judgment in relevant_judgments}
    if not gold_page_ids:
        return None
    if judgment_structure is JudgmentStructure.UNKNOWN:
        return (
            float(next(iter(gold_page_ids)) in observed_page_ids)
            if len(gold_page_ids) == 1
            else None
        )
    if judgment_structure in {JudgmentStructure.SINGLE, JudgmentStructure.ANY_OF}:
        if judgment_structure is JudgmentStructure.SINGLE and len(gold_page_ids) != 1:
            return None
        return float(bool(observed_page_ids & gold_page_ids))
    if judgment_structure is JudgmentStructure.ALL_OF:
        return len(observed_page_ids & gold_page_ids) / len(gold_page_ids)
    if judgment_structure is not JudgmentStructure.GROUPED or not requirement_groups:
        return None

    target_by_judgment = {
        judgment.judgment_id: judgment.target_id for judgment in relevant_judgments
    }
    group_scores: list[float] = []
    for group in requirement_groups:
        try:
            targets = {target_by_judgment[judgment_id] for judgment_id in group.judgment_ids}
        except KeyError:
            return None
        if group.mode == "any_of":
            group_scores.append(float(bool(observed_page_ids & targets)))
        else:
            group_scores.append(len(observed_page_ids & targets) / len(targets))
    return sum(group_scores) / len(group_scores)
