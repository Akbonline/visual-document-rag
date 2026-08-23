from pathlib import Path
from typing import Any

from vidore_rag.datasets import ViDoReV3Adapter
from vidore_rag.ingestion.materialize import (
    load_judgments,
    load_materialized_pages,
    load_queries,
    materialize_vidore,
)


class FakeImage:
    mode = "RGB"
    size = (100, 200)

    def save(self, path: Path, **_: Any) -> None:
        path.write_bytes(b"fake-jpeg-content")


def test_materialization_is_validated_and_reused(tmp_path: Path) -> None:
    adapter = ViDoReV3Adapter(
        corpus=[
            {
                "corpus_id": 7,
                "image": FakeImage(),
                "doc_id": "manual",
                "markdown": "Battery safety instructions.",
                "page_number_in_doc": 0,
            }
        ],
        queries=[
            {
                "query_id": 10,
                "query": "What are the safety instructions?",
                "language": "english",
                "answer": "Follow the battery instructions.",
            }
        ],
        qrels=[
            {
                "query_id": 10,
                "corpus_id": 7,
                "score": 2,
                "content_type": ["Text"],
                "bounding_boxes": [],
            }
        ],
    )

    first = materialize_vidore(adapter, tmp_path)
    second = materialize_vidore(adapter, tmp_path)

    assert not first.reused
    assert second.reused
    assert first.manifest.page_count == 1
    manifest_path = Path(first.manifest_path)
    assert len(load_materialized_pages(manifest_path)) == 1
    assert len(load_queries(manifest_path)) == 1
    assert len(load_judgments(manifest_path)) == 1
