from __future__ import annotations

import math
import re
from collections import Counter

from vidore_rag.document_ir import ChunkRecord, SearchHit

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Small in-memory BM25 baseline suitable for the weekend corpus slice."""

    def __init__(self, chunks: list[ChunkRecord], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(counts.values()) for counts in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_frequencies:
            self._document_frequency.update(counts.keys())

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        terms = set(tokenize(query))
        scored: list[tuple[float, ChunkRecord]] = []
        for chunk, counts, length in zip(
            self._chunks, self._term_frequencies, self._lengths, strict=True
        ):
            score = sum(self._term_score(term, counts[term], length) for term in terms)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchHit(
                candidate_id=chunk.chunk_id,
                score=score,
                rank=rank,
                source="bm25",
                page_id=chunk.page_id,
                document_id=chunk.document_id,
                text=chunk.text,
            )
            for rank, (score, chunk) in enumerate(scored[:limit], start=1)
        ]

    def _term_score(self, term: str, frequency: int, length: int) -> float:
        if frequency == 0:
            return 0.0
        document_count = len(self._chunks)
        document_frequency = self._document_frequency[term]
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        normalization = self._k1 * (
            1 - self._b + self._b * length / self._average_length
        )
        return inverse_document_frequency * (frequency * (self._k1 + 1)) / (
            frequency + normalization
        )
