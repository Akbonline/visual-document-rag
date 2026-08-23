from vidore_rag.datasets import ViDoReV3Adapter


def adapter() -> ViDoReV3Adapter:
    corpus = [
        {
            "corpus_id": 7,
            "image": {"width": 1654, "height": 2339},
            "doc_id": "manual",
            "markdown": "# Safety\nStop charging when the battery overheats.",
            "page_number_in_doc": 0,
        }
    ]
    queries = [
        {
            "query_id": 10,
            "query": "What should I do?",
            "language": "english",
            "answer": "Stop charging.",
        },
        {
            "query_id": 11,
            "query": "Que faire ?",
            "language": "french",
            "answer": "Arrêter la charge.",
        },
    ]
    qrels = [
        {
            "query_id": 10,
            "corpus_id": 7,
            "score": 2,
            "content_type": ["Text"],
            "bounding_boxes": [
                {"annotator": 0, "x1": 100, "x2": 900, "y1": 200, "y2": 400},
                {"annotator": 0, "x1": 822, "x2": 822, "y1": 200, "y2": 400},
            ],
        },
        {
            "query_id": 11,
            "corpus_id": 7,
            "score": 2,
            "content_type": ["Text"],
            "bounding_boxes": [],
        },
    ]
    return ViDoReV3Adapter(corpus=corpus, queries=queries, qrels=qrels)


def test_adapter_filters_language_and_preserves_answers() -> None:
    queries = list(adapter().iter_queries())

    assert len(queries) == 1
    assert queries[0].reference_answers == ["Stop charging."]
    assert queries[0].metadata["native_query_id"] == 10


def test_adapter_preserves_page_identity_geometry_and_text_source() -> None:
    page = next(iter(adapter().iter_pages()))

    assert page.page_id == "vidore/vidore_v3_hr:page:7"
    assert page.page_number == 1
    assert (page.width, page.height, page.dpi) == (1654, 2339, None)
    assert page.metadata["text_source"] == "shipped_markdown"


def test_adapter_maps_graded_page_judgments_and_pixel_boxes() -> None:
    judgment = next(iter(adapter().iter_judgments()))

    assert judgment.query_id == "vidore/vidore_v3_hr:query:10"
    assert judgment.target_id == "vidore/vidore_v3_hr:page:7"
    assert judgment.relevance == 2
    assert judgment.bounding_boxes[0].coordinate_space == "pixel"
    assert judgment.bounding_boxes[0].x2 == 900
    assert judgment.invalid_bounding_box_count == 1
