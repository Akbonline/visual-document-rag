from __future__ import annotations

import re
from collections import Counter

from vidore_rag.context import EvidenceBundle
from vidore_rag.contracts import BoundingBoxEvidence, JudgmentRecord
from vidore_rag.generation.models import (
    AnswerQuality,
    CitationQuality,
    CostMeasurement,
    GeneratedAnswer,
    LocalizationQuality,
    PredictedBoundingBox,
    ProviderConfig,
    TokenUsage,
)

TOKEN_PATTERN = re.compile(r"\w+")


def normalize_answer(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def score_answer(answer: str, references: list[str]) -> AnswerQuality:
    if not references:
        return AnswerQuality(exact_match=0, token_f1=0)
    normalized = normalize_answer(answer)
    return AnswerQuality(
        exact_match=max(
            float(normalized == normalize_answer(reference)) for reference in references
        ),
        token_f1=max(
            _token_f1(normalized, normalize_answer(reference)) for reference in references
        ),
    )


def score_citations(
    answer: GeneratedAnswer,
    evidence: EvidenceBundle,
    gold_page_ids: set[str],
) -> CitationQuality:
    citations = answer.citations
    if not citations:
        return CitationQuality(
            valid_context_precision=0,
            gold_page_precision=0,
            gold_page_recall=0,
            quote_support_precision=0,
        )
    context_by_page = {item.page_id: item.text for item in evidence.items}
    cited_pages = {citation.page_id for citation in citations}
    valid = sum(citation.page_id in context_by_page for citation in citations)
    gold = sum(citation.page_id in gold_page_ids for citation in citations)
    supported = sum(
        citation.page_id in context_by_page
        and normalize_answer(citation.quote) in normalize_answer(context_by_page[citation.page_id])
        for citation in citations
    )
    return CitationQuality(
        valid_context_precision=valid / len(citations),
        gold_page_precision=gold / len(citations),
        gold_page_recall=(
            len(cited_pages & gold_page_ids) / len(gold_page_ids) if gold_page_ids else 0
        ),
        quote_support_precision=supported / len(citations),
    )


def score_localization(
    answer: GeneratedAnswer,
    judgments: list[JudgmentRecord],
) -> LocalizationQuality:
    gold_by_page = {
        judgment.target_id: judgment.bounding_boxes
        for judgment in judgments
        if judgment.bounding_boxes
    }
    predicted = [
        (citation.page_id, citation.bounding_box)
        for citation in answer.citations
        if citation.bounding_box is not None
    ]
    gold_count = sum(len(boxes) for boxes in gold_by_page.values())
    if not predicted:
        return LocalizationQuality(
            status="not_applicable",
            predicted_box_count=0,
            gold_box_count=gold_count,
            reason="the text-only generation path did not predict pixel boxes",
        )
    if not gold_by_page:
        return LocalizationQuality(
            status="not_applicable",
            predicted_box_count=len(predicted),
            gold_box_count=0,
            reason="the dataset has no gold pixel boxes for this query",
        )
    scores = [
        max(
            (_iou(box, gold) for gold in gold_by_page.get(page_id, [])),
            default=0.0,
        )
        for page_id, box in predicted
        if box is not None
    ]
    return LocalizationQuality(
        status="applicable",
        mean_best_iou=sum(scores) / len(scores),
        predicted_box_count=len(predicted),
        gold_box_count=gold_count,
    )


def estimate_cost(usage: TokenUsage, config: ProviderConfig) -> CostMeasurement:
    if config.input_cost_per_million is None:
        return CostMeasurement(status="not_applicable", reason="token prices were not declared")
    assert config.cached_input_cost_per_million is not None
    assert config.output_cost_per_million is not None
    uncached_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    cost = (
        uncached_input * config.input_cost_per_million
        + usage.cached_input_tokens * config.cached_input_cost_per_million
        + usage.output_tokens * config.output_cost_per_million
    ) / 1_000_000
    return CostMeasurement(status="applicable", estimated_usd=cost)


def _token_f1(left: str, right: str) -> float:
    left_tokens = left.split()
    right_tokens = right.split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def _iou(predicted: PredictedBoundingBox, gold: BoundingBoxEvidence) -> float:
    intersection_width = max(0, min(predicted.x2, gold.x2) - max(predicted.x1, gold.x1))
    intersection_height = max(0, min(predicted.y2, gold.y2) - max(predicted.y1, gold.y1))
    intersection = intersection_width * intersection_height
    predicted_area = (predicted.x2 - predicted.x1) * (predicted.y2 - predicted.y1)
    gold_area = (gold.x2 - gold.x1) * (gold.y2 - gold.y1)
    union = predicted_area + gold_area - intersection
    return intersection / union if union else 0.0
