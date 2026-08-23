from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vidore_rag.context import ContextBuilder, EvidenceBundle
from vidore_rag.contracts import QueryRecord
from vidore_rag.document_ir import SearchHit
from vidore_rag.evaluation import BenchmarkSession
from vidore_rag.evaluation.attribution import (
    FailureClass,
    classify_failure,
    evidence_coverage,
    resolve_answerability,
)
from vidore_rag.generation.evaluation import (
    estimate_cost,
    score_answer,
    score_citations,
    score_localization,
)
from vidore_rag.generation.models import (
    GeneratedAnswer,
    GenerationConfig,
    GenerationMeasurement,
    GenerationMode,
    GenerationPairResult,
)
from vidore_rag.generation.prompting import build_request
from vidore_rag.generation.provider import LLMProvider
from vidore_rag.ingestion.materialize import load_judgments, load_materialized_pages


class GenerationRunner:
    def __init__(
        self,
        session: BenchmarkSession,
        provider: LLMProvider,
        config: GenerationConfig,
    ) -> None:
        if provider.name != config.provider.provider:
            raise ValueError("provider implementation does not match generation config")
        if provider.model != config.provider.model:
            raise ValueError("provider model does not match generation config")
        self.session = session
        self.provider = provider
        self.config = config
        source_manifest = Path(session.text_manifest.source_manifest_path)
        self.pages = {
            row.page.page_id: row.page for row in load_materialized_pages(source_manifest)
        }
        self.judgments = load_judgments(source_manifest)

    def run_pair(self, native_query_id: int) -> GenerationPairResult:
        retrieval = self.session.query_by_native_id(
            native_query_id, limit=self.config.top_k
        )
        threshold = self.session.capabilities.binary_relevance_threshold
        if threshold is None:
            raise ValueError("generation evaluation requires a binary relevance threshold")
        gold_page_ids = {
            page_id for page_id, relevance in retrieval.gold_pages.items() if relevance >= threshold
        }
        retrieved_bundle = ContextBuilder(
            token_budget=self.config.context_token_budget
        ).build(retrieval.hits)
        oracle_bundle = ContextBuilder(
            token_budget=self.config.context_token_budget
        ).build(self._oracle_hits(retrieval.query.query_id))
        retrieved = self._generate(
            mode=GenerationMode.RETRIEVED,
            query=retrieval.query,
            evidence=retrieved_bundle,
            retrieved_page_ids={hit.page_id for hit in retrieval.hits},
            gold_page_ids=gold_page_ids,
        )
        oracle = self._generate(
            mode=GenerationMode.ORACLE,
            query=retrieval.query,
            evidence=oracle_bundle,
            retrieved_page_ids=gold_page_ids,
            gold_page_ids=gold_page_ids,
        )
        return GenerationPairResult(
            query_id=retrieval.query.query_id,
            native_query_id=native_query_id,
            retrieved=retrieved,
            oracle=oracle,
        )

    def _generate(
        self,
        *,
        mode: GenerationMode,
        query: QueryRecord,
        evidence: EvidenceBundle,
        retrieved_page_ids: set[str],
        gold_page_ids: set[str],
    ) -> GenerationMeasurement:
        request = build_request(
            question=query.text,
            evidence=evidence,
            max_output_tokens=self.config.provider.max_output_tokens,
        )
        prompt_fingerprint = hashlib.sha256(
            json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()
        response = self.provider.generate(request)
        answer = GeneratedAnswer.model_validate_json(response.output_text)
        query_judgments = [
            judgment
            for judgment in self.judgments
            if judgment.query_id == query.query_id
        ]
        threshold = self.session.capabilities.binary_relevance_threshold
        assert threshold is not None
        relevant_judgments = [
            judgment for judgment in query_judgments if judgment.relevance >= threshold
        ]
        answer_quality = score_answer(answer.answer, query.reference_answers)
        citation_quality = score_citations(answer, evidence, gold_page_ids)
        localization_quality = score_localization(answer, query_judgments)
        context_page_ids = {item.page_id for item in evidence.items}
        retrieved_coverage = evidence_coverage(
            observed_page_ids=retrieved_page_ids,
            relevant_judgments=relevant_judgments,
            judgment_structure=self.session.capabilities.judgment_structure,
            requirement_groups=query.requirement_groups,
        )
        context_coverage = evidence_coverage(
            observed_page_ids=context_page_ids,
            relevant_judgments=relevant_judgments,
            judgment_structure=self.session.capabilities.judgment_structure,
            requirement_groups=query.requirement_groups,
        )
        answerability = resolve_answerability(
            is_answerable=query.is_answerable,
            has_answerability_label=self.session.capabilities.has_answerability_label,
            has_reference_answer=bool(query.reference_answers),
        )
        failure = classify_failure(
            gold_known=bool(gold_page_ids and query.reference_answers),
            answerability=answerability,
            system_refused=answer.refused,
            retrieved_coverage=retrieved_coverage,
            context_coverage=context_coverage,
            answer_correct=(
                answer_quality.token_f1 >= self.config.answer_token_f1_threshold
            ),
            citation_correct=(
                citation_quality.gold_page_recall == 1
                and citation_quality.quote_support_precision == 1
            ),
        )
        if (
            failure is FailureClass.SUCCESS
            and response.time_to_response_ms > self.config.latency_budget_ms
        ):
            failure = FailureClass.PERFORMANCE_FAILURE
        return GenerationMeasurement(
            query_id=query.query_id,
            native_query_id=int(query.metadata["native_query_id"]),
            mode=mode,
            answer=answer,
            reference_answers=query.reference_answers,
            context_page_ids=[item.page_id for item in evidence.items],
            gold_page_ids=sorted(gold_page_ids),
            answer_quality=answer_quality,
            citation_quality=citation_quality,
            localization_quality=localization_quality,
            usage=response.usage,
            cost=estimate_cost(response.usage, self.config.provider),
            latency_ms=response.latency_ms,
            time_to_response_ms=response.time_to_response_ms,
            provider_response_id=response.response_id,
            provider=response.provider,
            model=response.model,
            prompt_fingerprint=prompt_fingerprint,
            failure_class=failure.value,
            failure_reason=(
                "multi-page evidence has unknown joint-versus-alternative semantics"
                if failure is FailureClass.UNSCORABLE_GOLD
                and len(gold_page_ids) > 1
                and self.session.capabilities.judgment_structure.value == "unknown"
                else None
            ),
        )

    def _oracle_hits(self, query_id: str) -> list[SearchHit]:
        threshold = self.session.capabilities.binary_relevance_threshold
        if threshold is None:
            raise ValueError("oracle construction requires a binary relevance threshold")
        judgments = sorted(
            (
                row
                for row in self.judgments
                if row.query_id == query_id and row.relevance >= threshold
            ),
            key=lambda row: (-row.relevance, row.target_id),
        )
        hits: list[SearchHit] = []
        for rank, judgment in enumerate(judgments, start=1):
            page = self.pages.get(judgment.target_id)
            if page is None:
                raise ValueError(f"gold page {judgment.target_id!r} was not materialized")
            hits.append(
                SearchHit(
                    candidate_id=page.page_id,
                    page_id=page.page_id,
                    document_id=page.document_id,
                    rank=rank,
                    score=judgment.relevance,
                    source="oracle:qrels",
                    text=page.text,
                )
            )
        return hits
