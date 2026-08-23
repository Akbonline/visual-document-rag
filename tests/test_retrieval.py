from vidore_rag.chunking import FixedTokenChunker
from vidore_rag.document_ir import PageRecord, SearchHit
from vidore_rag.retrieval import BM25Index, project_chunks_to_pages, reciprocal_rank_fusion


def test_bm25_returns_relevant_page_first() -> None:
    pages = [
        PageRecord(
            page_id="manual:1",
            document_id="manual",
            page_number=1,
            text="Charge the battery before first use.",
        ),
        PageRecord(
            page_id="manual:2",
            document_id="manual",
            page_number=2,
            text="ERR-42 indicates unsafe battery temperature. Stop charging.",
        ),
    ]
    chunks = FixedTokenChunker(chunk_size=8, overlap=2).chunk_pages(pages)

    hits = BM25Index(chunks).search("What does ERR-42 mean?")
    pages = project_chunks_to_pages(hits)

    assert pages[0].page_id == "manual:2"
    assert pages[0].source.endswith("chunk_to_page:best_rank")


def test_rrf_rewards_candidates_seen_by_multiple_retrievers() -> None:
    shared = SearchHit(
        candidate_id="page:shared",
        score=0.1,
        rank=2,
        source="dense",
        page_id="page:shared",
        document_id="doc",
        text="shared evidence",
    )
    sparse = [
        SearchHit(
            candidate_id="page:sparse",
            score=10,
            rank=1,
            source="bm25",
            page_id="page:sparse",
            document_id="doc",
            text="sparse only",
        ),
        shared.model_copy(update={"source": "bm25"}),
    ]
    dense = [
        SearchHit(
            candidate_id="page:dense",
            score=0.9,
            rank=1,
            source="dense",
            page_id="page:dense",
            document_id="doc",
            text="dense only",
        ),
        shared,
    ]

    fused = reciprocal_rank_fusion({"sparse": sparse, "dense": dense})

    assert fused[0].candidate_id == "page:shared"
