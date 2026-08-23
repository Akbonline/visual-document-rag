from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vidore_rag.artifacts import read_json
from vidore_rag.contracts import QueryRecord
from vidore_rag.document_ir import SearchHit
from vidore_rag.evaluation.metrics import (
    AggregateRetrievalMetrics,
    QueryRetrievalMetrics,
    aggregate_metrics,
    judgments_by_query,
    score_query,
)
from vidore_rag.ingestion.materialize import load_judgments, load_queries
from vidore_rag.retrieval import BM25Index, project_chunks_to_pages, reciprocal_rank_fusion
from vidore_rag.retrieval.dense import (
    DenseIndex,
    DenseIndexManifest,
    SentenceTransformerEncoder,
    load_dense_index,
)
from vidore_rag.retrieval.indexing import TextIndexManifest, load_chunks

RetrieverMode = Literal["bm25", "dense", "hybrid"]


class QueryBenchmarkResult(BaseModel):
    query: QueryRecord
    mode: RetrieverMode
    hits: list[SearchHit]
    gold_pages: dict[str, float]
    metrics: QueryRetrievalMetrics


class BenchmarkReport(BaseModel):
    mode: RetrieverMode
    text_index_fingerprint: str
    dense_index_fingerprint: str | None = None
    aggregate: AggregateRetrievalMetrics
    queries: list[QueryRetrievalMetrics]


class BenchmarkSession:
    def __init__(
        self,
        text_index_manifest_path: Path,
        *,
        mode: RetrieverMode,
        dense_manifest_path: Path | None = None,
    ) -> None:
        self.text_manifest_path = text_index_manifest_path
        self.text_manifest = TextIndexManifest.model_validate(read_json(text_index_manifest_path))
        self.mode = mode
        self.chunks = load_chunks(text_index_manifest_path)
        self.bm25 = BM25Index(self.chunks)
        self.dense_manifest: DenseIndexManifest | None = None
        self.dense_index: DenseIndex | None = None
        self.encoder: SentenceTransformerEncoder | None = None
        if mode in {"dense", "hybrid"}:
            if dense_manifest_path is None:
                raise ValueError(f"{mode} mode requires a dense index manifest")
            self.dense_manifest, self.dense_index = load_dense_index(
                dense_manifest_path, text_index_manifest_path
            )
            self.encoder = SentenceTransformerEncoder(
                model_name=self.dense_manifest.model_name,
                revision=self.dense_manifest.model_revision,
                # Runtime hardware can differ from the machine that built the
                # portable embedding matrix, so resolve it again at query time.
                device="auto",
                local_files_only=True,
            )

        source_manifest_path = Path(self.text_manifest.source_manifest_path)
        self.queries = load_queries(source_manifest_path)
        self.relevance = judgments_by_query(load_judgments(source_manifest_path))

    def query_by_native_id(self, native_query_id: int, *, limit: int = 10) -> QueryBenchmarkResult:
        query = next(
            (
                candidate
                for candidate in self.queries
                if candidate.metadata.get("native_query_id") == native_query_id
            ),
            None,
        )
        if query is None:
            raise KeyError(f"native query ID {native_query_id} is not in the selected slice")
        return self.query(query, limit=limit)

    def query(self, query: QueryRecord, *, limit: int = 10) -> QueryBenchmarkResult:
        started = time.perf_counter()
        hits = self.search(query.text, limit=limit)
        latency_ms = (time.perf_counter() - started) * 1000
        relevance = self.relevance.get(query.query_id)
        if not relevance:
            raise ValueError(f"query {query.query_id} has no judgments")
        metrics = score_query(
            query_id=query.query_id,
            ranked_page_ids=[hit.page_id for hit in hits],
            relevance=relevance,
            latency_ms=latency_ms,
        )
        return QueryBenchmarkResult(
            query=query,
            mode=self.mode,
            hits=hits,
            gold_pages=relevance,
            metrics=metrics,
        )

    def search(self, question: str, *, limit: int = 10) -> list[SearchHit]:
        candidate_limit = max(limit * 5, 50)
        sparse_pages: list[SearchHit] = []
        dense_pages: list[SearchHit] = []
        if self.mode in {"bm25", "hybrid"}:
            sparse_chunks = self.bm25.search(question, limit=candidate_limit)
            sparse_pages = project_chunks_to_pages(sparse_chunks, limit=candidate_limit)
        if self.mode in {"dense", "hybrid"}:
            assert self.encoder is not None and self.dense_index is not None
            query_vector = self.encoder.encode([question], batch_size=1)[0]
            dense_chunks = self.dense_index.search_vector(
                query_vector, limit=candidate_limit
            )
            dense_pages = project_chunks_to_pages(dense_chunks, limit=candidate_limit)
        if self.mode == "bm25":
            return sparse_pages[:limit]
        if self.mode == "dense":
            return dense_pages[:limit]
        return reciprocal_rank_fusion(
            {"bm25": sparse_pages, "dense": dense_pages}, limit=limit
        )

    def evaluate(self, *, limit_queries: int | None = None, top_k: int = 10) -> BenchmarkReport:
        selected = self.queries[:limit_queries] if limit_queries is not None else self.queries
        rows = [self.query(query, limit=top_k).metrics for query in selected]
        return BenchmarkReport(
            mode=self.mode,
            text_index_fingerprint=self.text_manifest.fingerprint,
            dense_index_fingerprint=(
                self.dense_manifest.fingerprint if self.dense_manifest else None
            ),
            aggregate=aggregate_metrics(rows),
            queries=rows,
        )
