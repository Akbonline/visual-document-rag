from __future__ import annotations

from vidore_rag.document_ir import SearchHit


def project_chunks_to_pages(hits: list[SearchHit], *, limit: int = 10) -> list[SearchHit]:
    """Project chunk candidates to the page judgment unit using best rank."""

    best_by_page: dict[str, SearchHit] = {}
    for hit in hits:
        current = best_by_page.get(hit.page_id)
        if current is None or hit.rank < current.rank:
            best_by_page[hit.page_id] = hit

    ordered = sorted(best_by_page.values(), key=lambda hit: (hit.rank, -hit.score, hit.page_id))
    return [
        SearchHit(
            candidate_id=hit.page_id,
            score=hit.score,
            rank=rank,
            source=f"{hit.source}:chunk_to_page:best_rank",
            page_id=hit.page_id,
            document_id=hit.document_id,
            text=hit.text,
        )
        for rank, hit in enumerate(ordered[:limit], start=1)
    ]
