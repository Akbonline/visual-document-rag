from __future__ import annotations

from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from vidore_rag.artifacts import read_json, write_json
from vidore_rag.document_ir import ChunkRecord, SearchHit
from vidore_rag.ingestion.fingerprints import stage_fingerprint
from vidore_rag.retrieval.indexing import TextIndexManifest, load_chunks


class DenseIndexManifest(BaseModel):
    schema_version: str = "1"
    text_index_manifest_path: str
    text_index_fingerprint: str = Field(min_length=64, max_length=64)
    fingerprint: str = Field(min_length=64, max_length=64)
    model_name: str
    model_revision: str
    device: str
    batch_size: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    dimensions: int = Field(ge=1)
    embeddings_path: str = "embeddings.npy"


class DenseIndexResult(BaseModel):
    manifest_path: str
    manifest: DenseIndexManifest
    reused: bool


class SentenceTransformerEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "install sentence-transformers to build or query the dense index"
            ) from exc
        resolved_device = _resolve_device(device)
        self.model_name = model_name
        self.revision = revision
        self.device = resolved_device
        # PyTorch's MPS backend can crash when one model is entered concurrently.
        # Serialize only inference; retrieval callers and provider requests may
        # still run concurrently around this narrow critical section.
        self._encode_lock = Lock()
        self._model: Any = sentence_transformers.SentenceTransformer(
            model_name,
            revision=revision,
            device=resolved_device,
            local_files_only=local_files_only,
        )

    def encode(self, texts: list[str], *, batch_size: int) -> NDArray[np.float32]:
        with self._encode_lock:
            values = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return np.asarray(values, dtype=np.float32)


class DenseIndex:
    def __init__(
        self, chunks: list[ChunkRecord], embeddings: NDArray[np.float32]
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("dense embeddings must be a two-dimensional matrix")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunk and embedding counts do not match")
        self._chunks = chunks
        self._embeddings = embeddings

    def search_vector(
        self, query_embedding: NDArray[np.float32], *, limit: int = 10
    ) -> list[SearchHit]:
        if query_embedding.ndim != 1:
            raise ValueError("query embedding must be one-dimensional")
        if query_embedding.shape[0] != self._embeddings.shape[1]:
            raise ValueError("query and corpus embedding dimensions do not match")
        if limit < 1:
            raise ValueError("limit must be positive")
        scores = self._embeddings @ query_embedding
        ordered = np.argsort(-scores, kind="stable")[:limit]
        return [
            SearchHit(
                candidate_id=self._chunks[int(index)].chunk_id,
                score=float(scores[int(index)]),
                rank=rank,
                source="dense",
                page_id=self._chunks[int(index)].page_id,
                document_id=self._chunks[int(index)].document_id,
                text=self._chunks[int(index)].text,
            )
            for rank, index in enumerate(ordered, start=1)
        ]


def build_dense_index(
    text_index_manifest_path: Path,
    output_root: Path,
    *,
    model_name: str,
    model_revision: str,
    batch_size: int = 32,
    device: str = "auto",
) -> DenseIndexResult:
    text_manifest = TextIndexManifest.model_validate(read_json(text_index_manifest_path))
    encoder = SentenceTransformerEncoder(
        model_name=model_name, revision=model_revision, device=device
    )
    fingerprint = stage_fingerprint(
        stage_name="dense_index",
        implementation_version="1",
        config={"batch_size": batch_size, "normalize_embeddings": True},
        upstream_fingerprints=[text_manifest.fingerprint],
        model_revision=f"{model_name}@{model_revision}",
    )
    stage_dir = output_root / fingerprint
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        manifest = DenseIndexManifest.model_validate(read_json(manifest_path))
        embeddings_path = stage_dir / manifest.embeddings_path
        if manifest.fingerprint == fingerprint and embeddings_path.is_file():
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if embeddings.shape == (manifest.chunk_count, manifest.dimensions):
                return DenseIndexResult(
                    manifest_path=str(manifest_path), manifest=manifest, reused=True
                )

    chunks = load_chunks(text_index_manifest_path)
    embeddings = encoder.encode([chunk.text for chunk in chunks], batch_size=batch_size)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError("embedding model returned an invalid corpus matrix")
    stage_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = stage_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    manifest = DenseIndexManifest(
        text_index_manifest_path=str(text_index_manifest_path),
        text_index_fingerprint=text_manifest.fingerprint,
        fingerprint=fingerprint,
        model_name=model_name,
        model_revision=model_revision,
        device=encoder.device,
        batch_size=batch_size,
        chunk_count=len(chunks),
        dimensions=int(embeddings.shape[1]),
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return DenseIndexResult(manifest_path=str(manifest_path), manifest=manifest, reused=False)


def load_dense_index(
    dense_manifest_path: Path, text_index_manifest_path: Path
) -> tuple[DenseIndexManifest, DenseIndex]:
    manifest = DenseIndexManifest.model_validate(read_json(dense_manifest_path))
    text_manifest = TextIndexManifest.model_validate(read_json(text_index_manifest_path))
    if manifest.text_index_fingerprint != text_manifest.fingerprint:
        raise ValueError("dense and text index fingerprints do not match")
    chunks = load_chunks(text_index_manifest_path)
    values = np.load(dense_manifest_path.parent / manifest.embeddings_path)
    embeddings = np.asarray(values, dtype=np.float32)
    return manifest, DenseIndex(chunks, embeddings)


def _resolve_device(requested: str) -> str:
    try:
        torch = import_module("torch")
    except ImportError:
        if requested != "auto" and requested != "cpu":
            raise RuntimeError(f"{requested} requires PyTorch") from None
        return "cpu"
    if requested == "mps" and not bool(torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch runtime")
    if requested == "cuda" and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch runtime")
    if requested != "auto":
        return requested
    if bool(torch.backends.mps.is_available()):
        return "mps"
    if bool(torch.cuda.is_available()):
        return "cuda"
    return "cpu"
