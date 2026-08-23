from pathlib import Path

import pytest

from vidore_rag.config import inspect_dataset_config
from vidore_rag.datasets import build_default_registry
from vidore_rag.evaluation import BenchmarkSession
from vidore_rag.ocr import compare_ocr_to_source

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED = ROOT / (
    ".artifacts/vidore_v3_hr/materialized/"
    "c30f10d4d97b89e4d04d762683aabaa18de86731ef2bbbc7f1d75f87f4a2ebe3/manifest.json"
)
OCR = ROOT / (
    ".artifacts/vidore_v3_hr/ocr/"
    "2fc3607df33b7a32c2caaa164af04378b568c2bfc92da76fa3eb35ef3a6ff388/manifest.json"
)
TEXT_INDEX = ROOT / (
    ".artifacts/vidore_v3_hr/text_indexes/"
    "bf971d36baf939425ff8062e1ef1c25e6aa13ad8188c23e37f495b26807134b1/manifest.json"
)


@pytest.mark.skipif(not TEXT_INDEX.exists(), reason="real ViDoRe artifacts are not installed")
def test_real_vidore_bm25_quality_regression() -> None:
    registry = build_default_registry()
    inspection = inspect_dataset_config(
        ROOT / "configs/datasets/vidore_v3.yaml",
        available_adapters=registry.names(),
    )
    assert inspection.config is not None
    session = BenchmarkSession(
        TEXT_INDEX,
        mode="bm25",
        capabilities=inspection.config.evaluation.capabilities,
    )

    aggregate = session.evaluate().aggregate

    assert aggregate.query_count == 318
    assert aggregate.scored_query_count == 318
    assert aggregate.not_applicable_query_count == 0
    assert aggregate.ndcg_at_10 == pytest.approx(0.48648329102740473)
    assert aggregate.recall_at_10 == pytest.approx(0.5321503275231272)


@pytest.mark.skipif(
    not MATERIALIZED.exists() or not OCR.exists(),
    reason="real ViDoRe OCR artifacts are not installed",
)
def test_real_vidore_ocr_consistency_regression() -> None:
    comparison = compare_ocr_to_source(MATERIALIZED, OCR)

    assert comparison.page_count == 1110
    assert comparison.token_f1 == pytest.approx(0.8990273607038866)
