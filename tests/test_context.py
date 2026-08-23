from vidore_rag.context import ContextBuilder
from vidore_rag.document_ir import SearchHit


def hit(candidate_id: str, rank: int, text: str) -> SearchHit:
    return SearchHit(
        candidate_id=candidate_id,
        score=1 / rank,
        rank=rank,
        source="bm25",
        page_id=f"page:{rank}",
        document_id="manual",
        text=text,
    )


def test_context_builder_obeys_budget_and_adds_citations() -> None:
    bundle = ContextBuilder(token_budget=5).build(
        [hit("a", 1, "one two three"), hit("b", 2, "four five six")]
    )

    assert bundle.used_tokens == 3
    assert bundle.truncated
    assert len(bundle.items) == 1
    assert "[manual page 1]" in bundle.rendered_context
