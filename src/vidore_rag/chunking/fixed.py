from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field, model_validator

from vidore_rag.document_ir import ChunkRecord, PageRecord


class FixedTokenChunker(BaseModel):
    """Deterministic whitespace-token baseline; not the final layout-aware chunker."""

    chunk_size: int = Field(default=80, ge=1)
    overlap: int = Field(default=16, ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> FixedTokenChunker:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self

    def chunk_page(self, page: PageRecord) -> list[ChunkRecord]:
        tokens = page.text.split()
        if not tokens:
            return []

        chunks: list[ChunkRecord] = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            text = " ".join(tokens[start:end])
            identity = f"{page.page_id}:{start}:{end}:{text}".encode()
            chunks.append(
                ChunkRecord(
                    chunk_id=hashlib.sha256(identity).hexdigest(),
                    page_id=page.page_id,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    text=text,
                    token_start=start,
                    token_end=end,
                )
            )
            if end == len(tokens):
                break
        return chunks

    def chunk_pages(self, pages: list[PageRecord]) -> list[ChunkRecord]:
        return [chunk for page in pages for chunk in self.chunk_page(page)]
