from vidore_rag.ingestion.fingerprints import stage_fingerprint


def test_fingerprint_is_deterministic_across_mapping_order() -> None:
    left = stage_fingerprint(
        stage_name="ocr",
        implementation_version="1",
        config={"dpi": 200, "language": "en"},
        model_revision="engine@abc123",
    )
    right = stage_fingerprint(
        stage_name="ocr",
        implementation_version="1",
        config={"language": "en", "dpi": 200},
        model_revision="engine@abc123",
    )

    assert left == right


def test_dpi_or_model_change_invalidates_parse_fingerprint() -> None:
    baseline = stage_fingerprint(
        stage_name="ocr",
        implementation_version="1",
        config={"dpi": 200},
        model_revision="engine@abc123",
    )
    changed_dpi = stage_fingerprint(
        stage_name="ocr",
        implementation_version="1",
        config={"dpi": 300},
        model_revision="engine@abc123",
    )
    changed_model = stage_fingerprint(
        stage_name="ocr",
        implementation_version="1",
        config={"dpi": 200},
        model_revision="engine@def456",
    )

    assert baseline != changed_dpi
    assert baseline != changed_model

