from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from vidore_rag.context import EvidenceBundle
from vidore_rag.generation.models import (
    AnswerQuality,
    CitationQuality,
    GeneratedAnswer,
    LocalizationQuality,
)

RetrievalMode = Literal["bm25", "dense", "hybrid"]
CheckStatus = Literal["pass", "warning", "fail", "not_applicable"]


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=10, ge=1, le=20)
    context_token_budget: int = Field(default=1200, ge=64, le=4000)


class TraceHit(BaseModel):
    candidate_id: str
    page_id: str
    document_id: str
    page_number: int | None
    score: float
    rank: int = Field(ge=1)
    source: str
    text: str


class RetrievalStages(BaseModel):
    sparse_chunks: list[TraceHit]
    dense_chunks: list[TraceHit]
    sparse_pages: list[TraceHit]
    dense_pages: list[TraceHit]
    final_pages: list[TraceHit]


class StageTimings(BaseModel):
    sparse_ms: float = Field(ge=0)
    dense_ms: float = Field(ge=0)
    projection_and_fusion_ms: float = Field(ge=0)
    context_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class ContractCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str


class RetrievalTraceResponse(BaseModel):
    schema_version: str = "1"
    trace_id: str
    question: str
    mode: RetrievalMode
    top_k: int
    dataset_id: str
    dataset_fingerprint: str
    text_index_fingerprint: str
    dense_index_fingerprint: str
    projection_fingerprint: str
    benchmark_query_id: str | None
    stages: RetrievalStages
    context: EvidenceBundle
    copyable_prompt: str
    timings: StageTimings
    contract_checks: list[ContractCheck]


class ValidationRequest(BaseModel):
    trace_id: str = Field(min_length=32, max_length=64)
    answer: GeneratedAnswer


class ValidationCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str


class BenchmarkValidation(BaseModel):
    status: Literal["applicable", "not_applicable"]
    reason: str | None = None
    query_id: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    answer_quality: AnswerQuality | None = None
    citation_quality: CitationQuality | None = None
    localization_quality: LocalizationQuality | None = None


class ValidationResponse(BaseModel):
    schema_version: str = "1"
    trace_id: str
    valid: bool
    checks: list[ValidationCheck]
    grounding: CitationQuality
    benchmark: BenchmarkValidation


class QuerySummary(BaseModel):
    query_id: str
    native_query_id: int
    question: str
    evidence_types: list[str]
    query_types: list[str]


class QueryListResponse(BaseModel):
    dataset_id: str
    queries: list[QuerySummary]


class PageResponse(BaseModel):
    page_id: str
    document_id: str
    page_number: int
    width: int | None
    height: int | None
    text: str


class HealthResponse(BaseModel):
    status: Literal["ready"]
    dataset_id: str
    chunk_count: int
    page_count: int
    query_count: int
    dense_encoder_status: Literal["lazy", "loaded"]
