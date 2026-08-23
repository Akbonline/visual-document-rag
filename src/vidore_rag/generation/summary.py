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


class PairedMetricComparison(BaseModel):
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)


class GenerationPolicyComparison(BaseModel):
    query_count: int = Field(ge=1)
    retrieved_context_changed_query_count: int = Field(ge=0)
    oracle_prompt_match_count: int = Field(ge=0)
    retrieved_answer_token_f1: PairedMetricComparison
    oracle_answer_token_f1: PairedMetricComparison
    noise_adjusted_answer_token_f1_delta: float
    retrieved_gold_page_recall: PairedMetricComparison
    retrieved_quote_support_precision: PairedMetricComparison
    retrieved_ttr_ms: PairedMetricComparison
    oracle_ttr_ms: PairedMetricComparison
    interpretation: str


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


def compare_generation_policies(
    baseline: list[GenerationPairResult],
    candidate: list[GenerationPairResult],
) -> GenerationPolicyComparison:
    """Compare two generation runs over exactly the same query IDs."""

    baseline_by_id = {pair.query_id: pair for pair in baseline}
    candidate_by_id = {pair.query_id: pair for pair in candidate}
    if not baseline_by_id or baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("generation comparisons require identical non-empty query sets")
    ordered_ids = sorted(baseline_by_id)
    baseline_pairs = [baseline_by_id[query_id] for query_id in ordered_ids]
    candidate_pairs = [candidate_by_id[query_id] for query_id in ordered_ids]

    retrieved_f1 = _compare_metric(
        [pair.retrieved.answer_quality.token_f1 for pair in baseline_pairs],
        [pair.retrieved.answer_quality.token_f1 for pair in candidate_pairs],
    )
    oracle_f1 = _compare_metric(
        [pair.oracle.answer_quality.token_f1 for pair in baseline_pairs],
        [pair.oracle.answer_quality.token_f1 for pair in candidate_pairs],
    )
    return GenerationPolicyComparison(
        query_count=len(ordered_ids),
        retrieved_context_changed_query_count=sum(
            baseline_pair.retrieved.context_page_ids != candidate_pair.retrieved.context_page_ids
            for baseline_pair, candidate_pair in zip(baseline_pairs, candidate_pairs, strict=True)
        ),
        oracle_prompt_match_count=sum(
            baseline_pair.oracle.prompt_fingerprint == candidate_pair.oracle.prompt_fingerprint
            for baseline_pair, candidate_pair in zip(baseline_pairs, candidate_pairs, strict=True)
        ),
        retrieved_answer_token_f1=retrieved_f1,
        oracle_answer_token_f1=oracle_f1,
        noise_adjusted_answer_token_f1_delta=(retrieved_f1.mean_delta - oracle_f1.mean_delta),
        retrieved_gold_page_recall=_compare_metric(
            [pair.retrieved.citation_quality.gold_page_recall for pair in baseline_pairs],
            [pair.retrieved.citation_quality.gold_page_recall for pair in candidate_pairs],
        ),
        retrieved_quote_support_precision=_compare_metric(
            [pair.retrieved.citation_quality.quote_support_precision for pair in baseline_pairs],
            [pair.retrieved.citation_quality.quote_support_precision for pair in candidate_pairs],
        ),
        retrieved_ttr_ms=_compare_metric(
            [pair.retrieved.time_to_response_ms for pair in baseline_pairs],
            [pair.retrieved.time_to_response_ms for pair in candidate_pairs],
            lower_is_better=True,
        ),
        oracle_ttr_ms=_compare_metric(
            [pair.oracle.time_to_response_ms for pair in baseline_pairs],
            [pair.oracle.time_to_response_ms for pair in candidate_pairs],
            lower_is_better=True,
        ),
        interpretation=(
            "The repeated oracle arm uses identical prompts and estimates run-to-run "
            "model variance. The noise-adjusted F1 delta subtracts oracle drift from "
            "the retrieved-arm delta; it is diagnostic, not a significance test."
        ),
    )


def _compare_metric(
    baseline: list[float],
    candidate: list[float],
    *,
    lower_is_better: bool = False,
) -> PairedMetricComparison:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired metrics require equal non-empty samples")
    deltas = [right - left for left, right in zip(baseline, candidate, strict=True)]
    epsilon = 1e-12
    better = [(-delta if lower_is_better else delta) for delta in deltas]
    return PairedMetricComparison(
        baseline_mean=statistics.fmean(baseline),
        candidate_mean=statistics.fmean(candidate),
        mean_delta=statistics.fmean(deltas),
        wins=sum(value > epsilon for value in better),
        ties=sum(abs(value) <= epsilon for value in better),
        losses=sum(value < -epsilon for value in better),
    )


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
