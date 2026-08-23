from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from vidore_rag.document_ir import ChunkRecord, PageRecord

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FIGURE_PATTERN = re.compile(r"^\s*(?:!\[.*?\]\(.*?\)|(?:figure|fig\.)\s*\d*)", re.I)
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")


@dataclass(frozen=True)
class MarkdownBlock:
    kind: str
    text: str
    heading_path: tuple[str, ...]


class StructureAwareChunker(BaseModel):
    """Markdown-aware policy that keeps tables, captions, and headings explicit."""

    max_tokens: int = Field(default=180, ge=16)
    paragraph_overlap: int = Field(default=24, ge=0)

    def chunk_page(self, page: PageRecord) -> list[ChunkRecord]:
        blocks = parse_markdown_blocks(page.text)
        chunks: list[ChunkRecord] = []
        cursor = 0
        for block_index, block in enumerate(blocks):
            block_chunks = self._chunk_block(page, block, block_index, cursor)
            chunks.extend(block_chunks)
            cursor += len(block.text.split())
        return chunks

    def chunk_pages(self, pages: list[PageRecord]) -> list[ChunkRecord]:
        return [chunk for page in pages for chunk in self.chunk_page(page)]

    def _chunk_block(
        self,
        page: PageRecord,
        block: MarkdownBlock,
        block_index: int,
        token_offset: int,
    ) -> list[ChunkRecord]:
        if block.kind == "table":
            pieces = _split_table(block.text, self.max_tokens)
        elif block.kind in {"heading", "figure"}:
            pieces = [block.text]
        else:
            pieces = _window_text(
                block.text,
                max_tokens=self.max_tokens,
                overlap=min(self.paragraph_overlap, self.max_tokens - 1),
            )

        chunks: list[ChunkRecord] = []
        local_cursor = token_offset
        for piece_index, piece in enumerate(pieces):
            token_count = len(piece.split())
            identity = (
                f"{page.page_id}:{block.kind}:{block_index}:{piece_index}:"
                f"{'/'.join(block.heading_path)}:{piece}"
            ).encode()
            chunks.append(
                ChunkRecord(
                    chunk_id=hashlib.sha256(identity).hexdigest(),
                    page_id=page.page_id,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    text=piece,
                    token_start=local_cursor,
                    token_end=local_cursor + token_count,
                    kind=block.kind,
                    heading_path=list(block.heading_path),
                    metadata={"block_index": block_index, "piece_index": piece_index},
                )
            )
            local_cursor += max(1, token_count - self.paragraph_overlap)
        return chunks


def parse_markdown_blocks(text: str) -> list[MarkdownBlock]:
    lines = text.splitlines()
    blocks: list[MarkdownBlock] = []
    headings: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue

        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            headings = headings[: level - 1]
            headings.append(title)
            blocks.append(MarkdownBlock("heading", line, tuple(headings)))
            index += 1
            continue

        if _is_table_start(lines, index):
            table_lines: list[str] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index].rstrip())
                index += 1
            blocks.append(MarkdownBlock("table", "\n".join(table_lines), tuple(headings)))
            continue

        if FIGURE_PATTERN.match(line):
            blocks.append(MarkdownBlock("figure", line, tuple(headings)))
            index += 1
            continue

        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if not candidate.strip():
                index += 1
                break
            if HEADING_PATTERN.match(candidate) or FIGURE_PATTERN.match(candidate):
                break
            if _is_table_start(lines, index):
                break
            paragraph_lines.append(candidate)
            index += 1
        blocks.append(
            MarkdownBlock("paragraph", "\n".join(paragraph_lines), tuple(headings))
        )
    return blocks


def _is_table_start(lines: list[str], index: int) -> bool:
    if "|" not in lines[index]:
        return False
    if index + 1 < len(lines) and TABLE_SEPARATOR_PATTERN.match(lines[index + 1]):
        return True
    return lines[index].strip().startswith("|") and lines[index].count("|") >= 2


def _split_table(text: str, max_tokens: int) -> list[str]:
    lines = text.splitlines()
    if len(text.split()) <= max_tokens or len(lines) <= 2:
        return [text]

    header_count = 2 if len(lines) > 1 and TABLE_SEPARATOR_PATTERN.match(lines[1]) else 1
    header = lines[:header_count]
    rows = lines[header_count:]
    pieces: list[str] = []
    current = list(header)
    for row in rows:
        candidate = "\n".join([*current, row])
        if len(candidate.split()) > max_tokens and len(current) > len(header):
            pieces.append("\n".join(current))
            current = [*header, row]
        else:
            current.append(row)
    if len(current) > len(header) or not pieces:
        pieces.append("\n".join(current))
    return pieces


def _window_text(text: str, *, max_tokens: int, overlap: int) -> list[str]:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    pieces: list[str] = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        end = min(start + max_tokens, len(tokens))
        pieces.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
    return pieces
