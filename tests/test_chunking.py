import pytest
from pydantic import ValidationError

from vidore_rag.chunking import FixedTokenChunker
from vidore_rag.document_ir import PageRecord


def test_chunker_is_deterministic_and_preserves_page_provenance() -> None:
    page = PageRecord(
        page_id="doc:1", document_id="doc", page_number=1, text="one two three four five"
    )
    chunker = FixedTokenChunker(chunk_size=3, overlap=1)

    first = chunker.chunk_page(page)
    second = chunker.chunk_page(page)

    assert first == second
    assert [chunk.text for chunk in first] == ["one two three", "three four five"]
    assert all(chunk.page_id == "doc:1" for chunk in first)


def test_chunker_rejects_non_advancing_window() -> None:
    with pytest.raises(ValidationError, match="overlap must be smaller"):
        FixedTokenChunker(chunk_size=3, overlap=3)
