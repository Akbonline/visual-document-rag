import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_published_benchmark_summary_is_complete_and_internally_consistent() -> None:
    payload = json.loads((ROOT / "examples/benchmark_summary.json").read_text())
    runs = {run["name"]: run for run in payload["runs"]}

    assert payload["dataset"]["query_count"] == 318
    assert len(runs) == 5
    assert all(len(run["text_index_fingerprint"]) == 64 for run in runs.values())
    assert all(0 < run["ndcg_at_10"] <= 1 for run in runs.values())

    sparse = runs["bm25_fixed_markdown"]
    dense = runs["dense_fixed_markdown"]
    hybrid = runs["hybrid_fixed_markdown"]
    assert hybrid["ndcg_at_10"] > sparse["ndcg_at_10"] > dense["ndcg_at_10"]
    assert (hybrid["ndcg_at_10"] / sparse["ndcg_at_10"] - 1) * 100 == pytest.approx(
        6.8, abs=0.05
    )
    assert (hybrid["recall_at_10"] / sparse["recall_at_10"] - 1) * 100 == pytest.approx(
        8.8, abs=0.05
    )
