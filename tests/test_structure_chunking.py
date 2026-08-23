from vidore_rag.chunking import StructureAwareChunker, parse_markdown_blocks
from vidore_rag.document_ir import PageRecord


def test_markdown_parser_types_headings_tables_figures_and_paragraphs() -> None:
    markdown = """# Safety

Battery guidance paragraph.

| Code | Action |
| --- | --- |
| ERR-42 | Stop charging |

![Battery diagram](battery.png)
"""

    blocks = parse_markdown_blocks(markdown)

    assert [block.kind for block in blocks] == [
        "heading",
        "paragraph",
        "table",
        "figure",
    ]
    assert blocks[2].heading_path == ("Safety",)


def test_large_table_chunks_repeat_headers() -> None:
    rows = "\n".join(f"| ERR-{index} | action number {index} |" for index in range(12))
    page = PageRecord(
        page_id="manual:1",
        document_id="manual",
        page_number=1,
        text=f"| Code | Action |\n| --- | --- |\n{rows}",
    )

    chunks = StructureAwareChunker(max_tokens=20, paragraph_overlap=0).chunk_page(page)

    assert len(chunks) > 1
    assert all(chunk.kind == "table" for chunk in chunks)
    assert all(chunk.text.startswith("| Code | Action |\n| --- | --- |") for chunk in chunks)
