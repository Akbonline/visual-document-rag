from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationMode(StrEnum):
    RETRIEVED = "retrieved"
    ORACLE = "oracle"


class ProviderConfig(BaseModel):
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    model: str = Field(min_length=1)
    api_key_env: str = Field(default="OPENAI_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    base_url: str | None = None
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_output_tokens: int = Field(default=256, ge=16, le=4096)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    cached_input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_prices(self) -> ProviderConfig:
        prices = (
            self.input_cost_per_million,
            self.cached_input_cost_per_million,
            self.output_cost_per_million,
        )
        if any(price is not None for price in prices) and any(price is None for price in prices):
            raise ValueError("either declare all token prices or leave every price unset")
        return self


class GenerationConfig(BaseModel):
    provider: ProviderConfig
    prompt_version: str = Field(default="rag-cited-v2", min_length=1)
    context_token_budget: int = Field(default=1200, ge=64, le=100_000)
    top_k: int = Field(default=10, ge=1, le=100)
    answer_token_f1_threshold: float = Field(default=0.8, ge=0, le=1)
    latency_budget_ms: float = Field(default=10_000, gt=0)
    concurrency: int = Field(default=2, ge=1, le=16)
    requests_per_second: float = Field(default=2, gt=0, le=100)


class ProviderRequest(BaseModel):
    instructions: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=1)
    output_schema: dict[str, Any]


class TokenUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(ge=0)


class ProviderResponse(BaseModel):
    provider: str
    model: str
    response_id: str
    output_text: str
    usage: TokenUsage
    latency_ms: float = Field(ge=0)
    time_to_response_ms: float = Field(ge=0)


class PredictedBoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_space: Literal["pixel"]
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> PredictedBoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bounding box maximums must exceed minimums")
        return self


class GeneratedCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    bounding_box: PredictedBoundingBox | None


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    refused: bool
    citations: list[GeneratedCitation]

    @model_validator(mode="after")
    def validate_refusal(self) -> GeneratedAnswer:
        if self.refused and self.citations:
            raise ValueError("a refusal cannot cite supporting evidence")
        if not self.refused and not self.answer.strip():
            raise ValueError("a non-refusal must include an answer")
        return self


class AnswerQuality(BaseModel):
    exact_match: float = Field(ge=0, le=1)
    token_f1: float = Field(ge=0, le=1)


class CitationQuality(BaseModel):
    valid_context_precision: float = Field(ge=0, le=1)
    gold_page_precision: float = Field(ge=0, le=1)
    gold_page_recall: float = Field(ge=0, le=1)
    quote_support_precision: float = Field(ge=0, le=1)


class LocalizationQuality(BaseModel):
    status: str
    mean_best_iou: float | None = Field(default=None, ge=0, le=1)
    predicted_box_count: int = Field(ge=0)
    gold_box_count: int = Field(ge=0)
    reason: str | None = None


class CostMeasurement(BaseModel):
    status: str
    estimated_usd: float | None = Field(default=None, ge=0)
    reason: str | None = None


class GenerationMeasurement(BaseModel):
    query_id: str
    native_query_id: int
    mode: GenerationMode
    answer: GeneratedAnswer
    reference_answers: list[str]
    context_page_ids: list[str]
    gold_page_ids: list[str]
    answer_quality: AnswerQuality
    citation_quality: CitationQuality
    localization_quality: LocalizationQuality
    usage: TokenUsage
    cost: CostMeasurement
    latency_ms: float
    time_to_response_ms: float
    provider_response_id: str
    provider: str
    model: str
    prompt_fingerprint: str = Field(min_length=64, max_length=64)
    failure_class: str


class GenerationPairResult(BaseModel):
    query_id: str
    native_query_id: int
    retrieved: GenerationMeasurement
    oracle: GenerationMeasurement


class GenerationJobFailure(BaseModel):
    native_query_id: int
    error_type: str
    message: str


class GenerationExperimentManifest(BaseModel):
    schema_version: str = "1"
    fingerprint: str = Field(min_length=64, max_length=64)
    dataset_fingerprint: str = Field(min_length=64, max_length=64)
    text_index_fingerprint: str = Field(min_length=64, max_length=64)
    dense_index_fingerprint: str | None = None
    projection_fingerprint: str = Field(min_length=64, max_length=64)
    config: GenerationConfig
    native_query_ids: list[int]
    completed_query_ids: list[int]
    failures: list[GenerationJobFailure]
    results_directory: str
