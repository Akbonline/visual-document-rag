from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from vidore_rag.contracts import EvaluationCapabilities, JudgmentRecord, JudgmentStructure
from vidore_rag.evaluation.attribution import FailureClass
from vidore_rag.generation.models import GenerationMeasurement, GenerationPairResult


class GenerationArmSummary(BaseModel):
    n_queries: int = Field(ge=1)
    refusal_count: int = Field(ge=0)
    mean_answer_token_f1: float = Field(ge=0, le=1)
    mean_valid_context_precision: float = Field(ge=0, le=1)
    mean_gold_page_recall: float = Field(ge=0, le=1)
    mean_quote_support_precision: float = Field(ge=0, le=1)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    ttr_p50_ms: float = Field(ge=0)
    ttr_p95_ms: float = Field(ge=0)
    failure_counts: dict[str, int]


class GenerationExperimentSummary(BaseModel):
    query_count: int = Field(ge=1)
    attribution_policy: str
    retrieved: GenerationArmSummary
    oracle: GenerationArmSummary
    total_estimated_cost_usd: float = Field(ge=0)


class GenerationEvidenceRow(BaseModel):
    evidence_type: str
    n_queries: int = Field(ge=1)
    retrieved: GenerationArmSummary
    oracle: GenerationArmSummary


class GenerationEvidenceBreakdown(BaseModel):
    assignment_policy: str
    rows: list[GenerationEvidenceRow]


def summarize_generation_experiment(
    results: list[GenerationPairResult],
    judgments: list[JudgmentRecord],
    capabilities: EvaluationCapabilities,
) -> GenerationExperimentSummary:
    if not results:
        raise ValueError("cannot summarize an empty generation experiment")
    threshold = capabilities.binary_relevance_threshold
    if threshold is None:
        raise ValueError("generation summary requires a binary relevance threshold")
    relevant_counts: dict[str, int] = defaultdict(int)
    for judgment in judgments:
        if judgment.relevance >= threshold:
            relevant_counts[judgment.query_id] += 1

    retrieved = _summarize_arm([pair.retrieved for pair in results], relevant_counts, capabilities)
    oracle = _summarize_arm([pair.oracle for pair in results], relevant_counts, capabilities)
    return GenerationExperimentSummary(
        query_count=len(results),
        attribution_policy=(
            "multi-page queries with unknown AND/OR judgment semantics are "
            "reported as unscorable_gold"
        ),
        retrieved=retrieved,
        oracle=oracle,
        total_estimated_cost_usd=(retrieved.estimated_cost_usd + oracle.estimated_cost_usd),
    )


def summarize_generation_by_evidence_type(
    results: list[GenerationPairResult],
    judgments: list[JudgmentRecord],
    capabilities: EvaluationCapabilities,
) -> GenerationEvidenceBreakdown:
    """Summarize mutually exclusive evidence strata for a generation slice."""

    if not results:
        raise ValueError("cannot summarize an empty generation experiment")
    threshold = capabilities.binary_relevance_threshold
    if threshold is None:
        raise ValueError("generation summary requires a binary relevance threshold")

    relevant_counts: dict[str, int] = defaultdict(int)
    content_types: dict[str, set[str]] = defaultdict(set)
    for judgment in judgments:
        if judgment.relevance < threshold:
            continue
        relevant_counts[judgment.query_id] += 1
        content_types[judgment.query_id].update(judgment.content_types)

    grouped: dict[str, list[GenerationPairResult]] = defaultdict(list)
    for pair in results:
        grouped[assign_evidence_stratum(content_types[pair.query_id])].append(pair)

    order = ("Text-only", "Table", "Chart", "Infographic", "Unknown")
    rows = []
    for evidence_type in order:
        pairs = grouped.get(evidence_type)
        if not pairs:
            continue
        rows.append(
            GenerationEvidenceRow(
                evidence_type=evidence_type,
                n_queries=len(pairs),
                retrieved=_summarize_arm(
                    [pair.retrieved for pair in pairs], relevant_counts, capabilities
                ),
                oracle=_summarize_arm(
                    [pair.oracle for pair in pairs], relevant_counts, capabilities
                ),
            )
        )
    return GenerationEvidenceBreakdown(
        assignment_policy=(
            "mutually exclusive priority: Infographic > Chart > Table > Text-only; "
            "only relevant judgments contribute labels"
        ),
        rows=rows,
    )


def assign_evidence_stratum(content_types: set[str]) -> str:
    normalized = {value.strip().lower() for value in content_types}
    for label in ("Infographic", "Chart", "Table"):
        if label.lower() in normalized:
            return label
    if "text" in normalized:
        return "Text-only"
    return "Unknown"


def _summarize_arm(
    measurements: list[GenerationMeasurement],
    relevant_counts: dict[str, int],
    capabilities: EvaluationCapabilities,
) -> GenerationArmSummary:
    failures = Counter(
        _corrected_failure(measurement, relevant_counts, capabilities)
        for measurement in measurements
    )
    ttr = sorted(measurement.time_to_response_ms for measurement in measurements)
    return GenerationArmSummary(
        n_queries=len(measurements),
        refusal_count=sum(measurement.answer.refused for measurement in measurements),
        mean_answer_token_f1=statistics.fmean(
            measurement.answer_quality.token_f1 for measurement in measurements
        ),
        mean_valid_context_precision=statistics.fmean(
            measurement.citation_quality.valid_context_precision for measurement in measurements
        ),
        mean_gold_page_recall=statistics.fmean(
            measurement.citation_quality.gold_page_recall for measurement in measurements
        ),
        mean_quote_support_precision=statistics.fmean(
            measurement.citation_quality.quote_support_precision for measurement in measurements
        ),
        input_tokens=sum(measurement.usage.input_tokens for measurement in measurements),
        cached_input_tokens=sum(
            measurement.usage.cached_input_tokens for measurement in measurements
        ),
        output_tokens=sum(measurement.usage.output_tokens for measurement in measurements),
        total_tokens=sum(measurement.usage.total_tokens for measurement in measurements),
        estimated_cost_usd=sum(measurement.cost.estimated_usd or 0 for measurement in measurements),
        ttr_p50_ms=_percentile(ttr, 0.50),
        ttr_p95_ms=_percentile(ttr, 0.95),
        failure_counts=dict(sorted(failures.items())),
    )


def _corrected_failure(
    measurement: GenerationMeasurement,
    relevant_counts: dict[str, int],
    capabilities: EvaluationCapabilities,
) -> str:
    return resolve_recorded_failure(
        recorded_failure=measurement.failure_class,
        relevant_page_count=relevant_counts.get(measurement.query_id, 0),
        judgment_structure=capabilities.judgment_structure,
    )


def resolve_recorded_failure(
    *,
    recorded_failure: str,
    relevant_page_count: int,
    judgment_structure: JudgmentStructure,
) -> str:
    if judgment_structure is JudgmentStructure.UNKNOWN and relevant_page_count > 1:
        return FailureClass.UNSCORABLE_GOLD.value
    return recorded_failure


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
