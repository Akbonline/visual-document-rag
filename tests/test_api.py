import asyncio
import json
from pathlib import Path

import httpx
import numpy as np

from vidore_rag.api.app import create_app
from vidore_rag.api.engine import RetrievalEngine
from vidore_rag.api.models import RetrieveRequest
from vidore_rag.artifacts import sha256_file
from vidore_rag.generation.models import GeneratedAnswer


class FakeEncoder:
    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        assert texts and batch_size == 1
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def artifact_bundle(tmp_path: Path) -> Path:
    chunks = [
        {
            "chunk_id": "manual:1:chunk:0",
            "page_id": "manual:page:1",
            "document_id": "manual",
            "page_number": 1,
            "text": "ERR-42 means the battery temperature is above its safe operating range.",
            "token_start": 0,
            "token_end": 11,
            "kind": "fixed",
            "heading_path": [],
            "metadata": {},
        },
        {
            "chunk_id": "manual:2:chunk:0",
            "page_id": "manual:page:2",
            "document_id": "manual",
            "page_number": 2,
            "text": "Restart the controller after inspecting its cooling fan.",
            "token_start": 0,
            "token_end": 8,
            "kind": "fixed",
            "heading_path": [],
            "metadata": {},
        },
    ]
    pages = [
        {
            "page_id": "manual:page:1",
            "document_id": "manual",
            "page_number": 1,
            "text": chunks[0]["text"],
            "image_uri": None,
            "width": 1000,
            "height": 1500,
            "dpi": None,
            "metadata": {},
        },
        {
            "page_id": "manual:page:2",
            "document_id": "manual",
            "page_number": 2,
            "text": chunks[1]["text"],
            "image_uri": None,
            "width": 1000,
            "height": 1500,
            "dpi": None,
            "metadata": {},
        },
    ]
    query = {
        "query_id": "manual:query:1",
        "text": "What does ERR-42 mean?",
        "is_answerable": True,
        "reference_answers": ["The battery temperature is above its safe operating range."],
        "requirement_groups": [],
        "metadata": {
            "native_query_id": 1,
            "content_type": ["Text"],
            "query_types": ["factoid"],
        },
    }
    judgment = {
        "judgment_id": "manual:judgment:1",
        "query_id": "manual:query:1",
        "target_id": "manual:page:1",
        "judgment_unit": "page",
        "relevance": 2,
        "content_types": ["Text"],
        "bounding_boxes": [],
        "invalid_bounding_box_count": 0,
    }
    write_jsonl(tmp_path / "chunks.jsonl", chunks)
    write_jsonl(tmp_path / "pages.jsonl", pages)
    write_jsonl(tmp_path / "queries.jsonl", [query])
    write_jsonl(tmp_path / "judgments.jsonl", [judgment])
    np.save(tmp_path / "embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    names = ("chunks.jsonl", "embeddings.npy", "pages.jsonl", "queries.jsonl", "judgments.jsonl")
    manifest = {
        "schema_version": "1",
        "dataset_id": "manual",
        "dataset_fingerprint": "a" * 64,
        "text_index_fingerprint": "b" * 64,
        "dense_index_fingerprint": "c" * 64,
        "projection_fingerprint": "d" * 64,
        "model_name": "test-encoder",
        "model_revision": "test-revision",
        "chunk_count": 2,
        "page_count": 2,
        "query_count": 1,
        "files": {name: sha256_file(tmp_path / name) for name in names},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def engine_for(tmp_path: Path) -> RetrievalEngine:
    return RetrievalEngine(
        artifact_bundle(tmp_path),
        encoder_factory=lambda _model, _revision: FakeEncoder(),
    )


def test_retrieval_trace_and_grounded_validation(tmp_path: Path) -> None:
    engine = engine_for(tmp_path)
    assert engine.health().dense_encoder_status == "lazy"
    engine.preload_dense()
    assert engine.health().dense_encoder_status == "loaded"
    trace = engine.retrieve(
        RetrieveRequest(
            question="What does ERR-42 mean?",
            mode="hybrid",
            top_k=2,
            context_token_budget=120,
        )
    )

    assert trace.benchmark_query_id == "manual:query:1"
    assert trace.stages.sparse_chunks[0].page_id == "manual:page:1"
    assert trace.stages.dense_chunks[0].page_id == "manual:page:1"
    assert trace.stages.final_pages[0].source == "rrf"
    assert "What does ERR-42 mean?" in trace.copyable_prompt
    assert all(check.status == "pass" for check in trace.contract_checks)

    answer = GeneratedAnswer.model_validate(
        {
            "answer": "The battery temperature is above its safe operating range.",
            "refused": False,
            "citations": [
                {
                    "page_id": "manual:page:1",
                    "quote": "battery temperature is above its safe operating range",
                    "bounding_box": None,
                }
            ],
        }
    )
    validation = engine.validate(trace.trace_id, answer)

    assert validation.valid
    assert validation.grounding.valid_context_precision == 1
    assert validation.grounding.quote_support_precision == 1
    assert validation.benchmark.status == "applicable"
    assert validation.benchmark.answer_quality is not None
    assert validation.benchmark.answer_quality.token_f1 == 1


def test_validator_rejects_unknown_page_and_unsupported_quote(tmp_path: Path) -> None:
    engine = engine_for(tmp_path)
    trace = engine.retrieve(
        RetrieveRequest(question="What does ERR-42 mean?", mode="bm25")
    )
    answer = GeneratedAnswer.model_validate(
        {
            "answer": "Invented answer",
            "refused": False,
            "citations": [
                {
                    "page_id": "manual:page:99",
                    "quote": "not in the evidence",
                    "bounding_box": None,
                }
            ],
        }
    )

    validation = engine.validate(trace.trace_id, answer)

    assert not validation.valid
    failed = {check.name for check in validation.checks if check.status == "fail"}
    assert failed == {"Citation page membership", "Quote support"}


def test_validator_accepts_refusal_without_citation_checks(tmp_path: Path) -> None:
    engine = engine_for(tmp_path)
    trace = engine.retrieve(
        RetrieveRequest(question="What fact is absent from these pages?", mode="bm25")
    )
    answer = GeneratedAnswer.model_validate(
        {
            "answer": "The supplied evidence does not state that fact.",
            "refused": True,
            "citations": [],
        }
    )

    validation = engine.validate(trace.trace_id, answer)

    assert validation.valid
    statuses = {check.name: check.status for check in validation.checks}
    assert statuses == {
        "Structured output": "pass",
        "Citation presence": "not_applicable",
        "Citation page membership": "not_applicable",
        "Quote support": "not_applicable",
        "Bounding box coordinate space": "not_applicable",
    }
    assert validation.benchmark.status == "not_applicable"


def test_fastapi_contract_exposes_health_retrieval_and_validation(tmp_path: Path) -> None:
    test_engine = engine_for(tmp_path)
    application = create_app(engine=test_engine)
    application.state.engine = test_engine

    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/api/v1/health")
            assert health.status_code == 200
            assert health.json()["chunk_count"] == 2

            retrieval = await client.post(
                "/api/v1/retrieve",
                json={"question": "What does ERR-42 mean?", "mode": "bm25"},
            )
            assert retrieval.status_code == 200
            trace_id = retrieval.json()["trace_id"]

            validation = await client.post(
                "/api/v1/validate",
                json={
                    "trace_id": trace_id,
                    "answer": {
                        "answer": "The battery is too hot.",
                        "refused": False,
                        "citations": [
                            {
                                "page_id": "manual:page:1",
                                "quote": "battery temperature is above its safe operating range",
                                "bounding_box": None,
                            }
                        ],
                    },
                },
            )
            assert validation.status_code == 200
            assert validation.json()["valid"] is True

    asyncio.run(exercise_api())
