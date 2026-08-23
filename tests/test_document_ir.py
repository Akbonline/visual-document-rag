import pytest
from pydantic import ValidationError

from vidore_rag.document_ir import PageRecord


def test_page_geometry_is_all_or_nothing() -> None:
    with pytest.raises(ValidationError, match="width, height, and dpi"):
        PageRecord(
            page_id="doc:1",
            document_id="doc",
            page_number=1,
            text="page",
            width=1200,
        )


def test_page_accepts_declared_coordinate_space() -> None:
    page = PageRecord(
        page_id="doc:1",
        document_id="doc",
        page_number=1,
        text="page",
        width=1200,
        height=1600,
        dpi=144,
    )
    assert page.dpi == 144
