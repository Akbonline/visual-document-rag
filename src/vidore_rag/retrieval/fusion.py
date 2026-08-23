from __future__ import annotations

from collections import defaultdict

from vidore_rag.document_ir import SearchHit


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[SearchHit]], *, k: int = 60, limit: int = 10
) -> list[SearchHit]:
    """Fuse heterogeneous rankings without assuming comparable raw scores."""

    if k < 1 or limit < 1:
        raise ValueError("k and limit must be positive")

    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, SearchHit] = {}
    for hits in ranked_lists.values():
        for hit in hits:
            scores[hit.candidate_id] += 1 / (k + hit.rank)
            representatives.setdefault(hit.candidate_id, hit)

    ordered_ids = sorted(scores, key=lambda candidate_id: (-scores[candidate_id], candidate_id))
    return [
        SearchHit(
            candidate_id=candidate_id,
            score=scores[candidate_id],
            rank=rank,
            source="rrf",
            page_id=representatives[candidate_id].page_id,
            document_id=representatives[candidate_id].document_id,
            text=representatives[candidate_id].text,
        )
        for rank, candidate_id in enumerate(ordered_ids[:limit], start=1)
    ]
