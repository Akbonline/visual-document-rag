from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PageRecord(BaseModel):
    """Canonical page-level record shared by every dataset adapter."""

    page_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str
    image_uri: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    dpi: float | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_geometry(self) -> PageRecord:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be declared together")
        if self.dpi is not None and self.width is None:
            raise ValueError("dpi requires declared width and height")
        return self


class ChunkRecord(BaseModel):
    """A retrieval unit with explicit provenance back to its source page."""

    chunk_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span(self) -> ChunkRecord:
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be greater than token_start")
        return self


class SearchHit(BaseModel):
    candidate_id: str = Field(min_length=1)
    score: float
    rank: int = Field(ge=1)
    source: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
