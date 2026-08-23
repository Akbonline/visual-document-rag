import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vidore_rag.contracts import (
    BoundingBoxEvidence,
    JudgmentRecord,
    JudgmentStructure,
    JudgmentUnit,
)
from vidore_rag.datasets.vidore_v3 import ViDoReV3Adapter
from vidore_rag.evaluation import BenchmarkSession
from vidore_rag.generation import (
    GeneratedAnswer,
    GeneratedCitation,
    GenerationConfig,
    GenerationRunner,
    LLMProvider,
    OpenAIResponsesProvider,
    PredictedBoundingBox,
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    TokenUsage,
    load_generation_results,
    resolve_recorded_failure,
    run_generation_experiment,
    score_localization,
    summarize_generation_by_evidence_type,
)
from vidore_rag.ingestion import materialize_dataset
from vidore_rag.retrieval import build_text_index


class FakeImage:
    mode = "RGB"
    size = (1000, 1200)

    def save(self, path: Path, **_: Any) -> None:
        path.write_bytes(b"generation-contract-image")


class RecordingProvider(LLMProvider):
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "recording"

    @property
    def model(self) -> str:
        return "recording-v1"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        page_id = "vidore/vidore_v3_hr:page:7"
        payload = GeneratedAnswer(
            answer="Stop charging.",
            refused=False,
            citations=[
                GeneratedCitation(
                    page_id=page_id,
                    quote="Stop charging",
                    bounding_box=None,
                )
            ],
        )
        return ProviderResponse(
            provider=self.name,
            model=self.model,
            response_id=f"response-{len(self.requests)}",
            output_text=payload.model_dump_json(),
            usage=TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110),
            latency_ms=25,
            time_to_response_ms=25,
        )


def test_retrieved_and_oracle_generation_use_the_shared_contract(tmp_path: Path) -> None:
    adapter = ViDoReV3Adapter(
        corpus=[
            {
                "corpus_id": 7,
                "doc_id": "manual",
                "page_number_in_doc": 0,
                "markdown": "Battery warning: Stop charging immediately.",
                "image": FakeImage(),
            },
            {
                "corpus_id": 8,
                "doc_id": "manual",
                "page_number_in_doc": 1,
                "markdown": "Routine cleaning instructions.",
                "image": FakeImage(),
            },
        ],
        queries=[
            {
                "query_id": 42,
                "query": "What should I do when the battery warning appears?",
                "answer": "Stop charging.",
                "language": "english",
                "query_types": [],
                "content_type": ["text"],
            }
        ],
        qrels=[
            {
                "query_id": 42,
                "corpus_id": 7,
                "score": 2,
                "content_type": ["text"],
                "bounding_boxes": [
                    {
                        "coordinate_space": "pixel",
                        "x1": 10,
                        "y1": 20,
                        "x2": 200,
                        "y2": 100,
                    }
                ],
            }
        ],
    )
    materialized = materialize_dataset(adapter, tmp_path / "materialized")
    indexed = build_text_index(
        Path(materialized.manifest_path),
        tmp_path / "indexes",
        policy="fixed",
        chunk_size=64,
        overlap=8,
    )
    session = BenchmarkSession(Path(indexed.manifest_path), mode="bm25")
    provider = RecordingProvider()
    config = GenerationConfig(
        provider=ProviderConfig(
            provider="recording",
            model="recording-v1",
            input_cost_per_million=1,
            cached_input_cost_per_million=0.1,
            output_cost_per_million=2,
        ),
        context_token_budget=64,
        top_k=2,
    )

    result = GenerationRunner(session, provider, config).run_pair(42)

    assert len(provider.requests) == 2
    assert provider.requests[0].instructions == provider.requests[1].instructions
    assert provider.requests[0].max_output_tokens == provider.requests[1].max_output_tokens
    assert result.retrieved.answer_quality.exact_match == 1
    assert result.oracle.answer_quality.exact_match == 1
    assert result.retrieved.citation_quality.quote_support_precision == 1
    assert result.retrieved.cost.estimated_usd == pytest.approx(0.00012)
    assert result.retrieved.failure_class == "success"
    assert result.retrieved.localization_quality.status == "not_applicable"

    breakdown = summarize_generation_by_evidence_type(
        [result], list(adapter.iter_judgments()), session.capabilities
    )
    assert [(row.evidence_type, row.n_queries) for row in breakdown.rows] == [("Text-only", 1)]
    assert breakdown.rows[0].retrieved.mean_answer_token_f1 == 1


def test_localization_scores_predicted_boxes_against_real_pixel_gold() -> None:
    page_id = "dataset:page:7"
    answer = GeneratedAnswer(
        answer="Stop charging.",
        refused=False,
        citations=[
            GeneratedCitation(
                page_id=page_id,
                quote="Stop charging",
                bounding_box=PredictedBoundingBox(
                    coordinate_space="pixel", x1=10, y1=20, x2=200, y2=100
                ),
            )
        ],
    )
    judgment = JudgmentRecord(
        judgment_id="j1",
        query_id="q1",
        target_id=page_id,
        judgment_unit=JudgmentUnit.PAGE,
        relevance=2,
        bounding_boxes=[BoundingBoxEvidence(x1=10, y1=20, x2=200, y2=100)],
    )

    result = score_localization(answer, [judgment])

    assert result.status == "applicable"
    assert result.mean_best_iou == 1


def test_provider_requires_api_key_without_leaking_a_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = ProviderConfig(provider="openai", model="gpt-5.4-mini")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY") as raised:
        OpenAIResponsesProvider(config)

    assert "sk-" not in str(raised.value)


def test_openai_provider_reports_output_budget_truncation_clearly() -> None:
    provider = object.__new__(OpenAIResponsesProvider)
    provider._config = ProviderConfig(provider="openai", model="gpt-5.4-mini")
    provider._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )
        )
    )
    request = ProviderRequest(
        instructions="Use the evidence.",
        prompt="Question and evidence",
        max_output_tokens=256,
        output_schema=GeneratedAnswer.model_json_schema(),
    )

    with pytest.raises(RuntimeError, match="truncated at max_output_tokens=256"):
        provider.generate(request)


def test_generation_experiment_resumes_completed_query_files(tmp_path: Path) -> None:
    adapter = ViDoReV3Adapter(
        corpus=[
            {
                "corpus_id": 7,
                "doc_id": "manual",
                "page_number_in_doc": 0,
                "markdown": "Battery warning: Stop charging immediately.",
                "image": FakeImage(),
            }
        ],
        queries=[
            {
                "query_id": 42,
                "query": "What should I do when the battery warning appears?",
                "answer": "Stop charging.",
                "language": "english",
                "query_types": [],
                "content_type": ["text"],
            }
        ],
        qrels=[
            {
                "query_id": 42,
                "corpus_id": 7,
                "score": 2,
                "content_type": ["text"],
                "bounding_boxes": [],
            }
        ],
    )
    materialized = materialize_dataset(adapter, tmp_path / "materialized")
    indexed = build_text_index(
        Path(materialized.manifest_path),
        tmp_path / "indexes",
        policy="fixed",
        chunk_size=64,
        overlap=8,
    )
    session = BenchmarkSession(Path(indexed.manifest_path), mode="bm25")
    provider = RecordingProvider()
    config = GenerationConfig(
        provider=ProviderConfig(provider="recording", model="recording-v1"),
        context_token_budget=64,
        requests_per_second=100,
    )
    runner = GenerationRunner(session, provider, config)
    progress: list[tuple[int, int, str]] = []

    first = run_generation_experiment(
        runner,
        provider,
        config,
        tmp_path / "generation",
        native_query_ids=[42],
        on_progress=lambda completed, total, message: progress.append((completed, total, message)),
    )
    second = run_generation_experiment(
        runner, provider, config, tmp_path / "generation", native_query_ids=[42]
    )
    manifest_path = tmp_path / "generation" / first.fingerprint / "manifest.json"

    assert first.completed_query_ids == [42]
    assert second.completed_query_ids == [42]
    assert len(provider.requests) == 2
    assert len(load_generation_results(manifest_path)) == 1
    assert progress == [(1, 1, "query 42 completed")]


def test_generated_answer_schema_forbids_unexpected_provider_fields() -> None:
    payload = {
        "answer": "Stop charging.",
        "refused": False,
        "citations": [],
        "chain_of_thought": "private reasoning",
    }

    with pytest.raises(ValueError):
        GeneratedAnswer.model_validate_json(json.dumps(payload))


def test_refusal_may_cite_inspected_evidence_for_transparency() -> None:
    answer = GeneratedAnswer(
        answer="The supplied evidence is insufficient.",
        refused=True,
        citations=[
            GeneratedCitation(
                page_id="dataset:page:7",
                quote="Available but incomplete evidence",
                bounding_box=None,
            )
        ],
    )

    assert answer.refused is True
    assert len(answer.citations) == 1


def test_generated_answer_schema_satisfies_strict_provider_requirements() -> None:
    schema = GeneratedAnswer.model_json_schema()
    citation_schema = schema["$defs"]["GeneratedCitation"]
    box_schema = schema["$defs"]["PredictedBoundingBox"]

    for object_schema in (schema, citation_schema, box_schema):
        assert object_schema["additionalProperties"] is False
        assert set(object_schema["required"]) == set(object_schema["properties"])


def test_recorded_attribution_is_unscorable_for_unknown_multi_page_gold() -> None:
    result = resolve_recorded_failure(
        recorded_failure="retrieval_incomplete",
        relevant_page_count=3,
        judgment_structure=JudgmentStructure.UNKNOWN,
    )

    assert result == "unscorable_gold"
