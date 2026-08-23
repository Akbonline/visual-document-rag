import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

from vidore_rag.document_ir import ChunkRecord
from vidore_rag.retrieval import dense
from vidore_rag.retrieval.dense import DenseIndex, SentenceTransformerEncoder


def chunk(chunk_id: str, page_id: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        page_id=page_id,
        document_id="manual",
        page_number=1,
        text=chunk_id,
        token_start=0,
        token_end=1,
    )


def test_dense_index_ranks_by_cosine_compatible_dot_product() -> None:
    index = DenseIndex(
        [chunk("battery", "page:1"), chunk("network", "page:2")],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    hits = index.search_vector(np.asarray([0.9, 0.1], dtype=np.float32))

    assert hits[0].candidate_id == "battery"
    assert hits[0].score == pytest.approx(0.9)


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(dense, "import_module", lambda _: fake_torch)

    assert dense._resolve_device("auto") == "cpu"


def test_explicit_unavailable_mps_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(dense, "import_module", lambda _: fake_torch)

    with pytest.raises(RuntimeError, match="MPS was requested but is unavailable"):
        dense._resolve_device("mps")


def test_sentence_transformer_encoder_serializes_shared_model_inference() -> None:
    class ConcurrentEntryDetector:
        def __init__(self) -> None:
            self.guard = Lock()
            self.active = 0
            self.maximum_active = 0

        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            with self.guard:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            time.sleep(0.02)
            with self.guard:
                self.active -= 1
            return np.ones((len(texts), 2), dtype=np.float32)

    model = ConcurrentEntryDetector()
    encoder = object.__new__(SentenceTransformerEncoder)
    encoder._model = model
    encoder._encode_lock = Lock()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(encoder.encode, [question], batch_size=1)
            for question in ("first", "second")
        ]
        for future in futures:
            assert future.result().shape == (1, 2)

    assert model.maximum_active == 1
