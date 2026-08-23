from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZATION = (
    ROOT
    / ".artifacts/vidore_v3_hr/materialized"
    / "c30f10d4d97b89e4d04d762683aabaa18de86731ef2bbbc7f1d75f87f4a2ebe3"
)
GATE_C_RUN = (
    ROOT
    / ".artifacts/vidore_v3_hr/generation"
    / "a2d9b0cf5b7a692f771cd984850e41f418c68018aeba6218258fe705577d0f63"
)
FRONTEND = ROOT / "frontend"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def document_title(document_id: str) -> str:
    slug = document_id.rsplit(":", 1)[-1].rsplit("-", 1)[0]
    return " ".join(word.capitalize() for word in slug.replace("_", " ").split())


def export_page(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.thumbnail((1000, 1450), Image.Resampling.LANCZOS)
        # Fast encoding keeps the full export suitable for a local release step.
        # The source resolution metadata is retained for accurate box overlays.
        image.save(destination, "WEBP", quality=70, method=0)


def public_arm(arm: dict[str, Any]) -> dict[str, Any]:
    """Keep reviewer-facing measurements while omitting provider identifiers."""

    fields = (
        "answer",
        "answer_quality",
        "citation_quality",
        "context_page_ids",
        "gold_page_ids",
        "cost",
        "failure_class",
        "failure_reason",
        "localization_quality",
        "model",
        "time_to_response_ms",
        "usage",
    )
    return {field: arm[field] for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the real Gate C evidence demo")
    parser.add_argument("--materialization", type=Path, default=MATERIALIZATION)
    parser.add_argument("--generation-run", type=Path, default=GATE_C_RUN)
    parser.add_argument("--frontend", type=Path, default=FRONTEND)
    args = parser.parse_args()

    queries = {row["query_id"]: row for row in read_jsonl(args.materialization / "queries.jsonl")}
    judgments_by_query: dict[str, list[dict[str, Any]]] = {}
    for judgment in read_jsonl(args.materialization / "judgments.jsonl"):
        judgments_by_query.setdefault(judgment["query_id"], []).append(judgment)
    pages = {
        row["page"]["page_id"]: row
        for row in read_jsonl(args.materialization / "pages.jsonl")
    }

    output_images = args.frontend / "public/evidence"
    if output_images.exists():
        shutil.rmtree(output_images)
    output_images.mkdir(parents=True)

    exported_pages: dict[str, dict[str, Any]] = {}
    exported_queries: list[dict[str, Any]] = []
    result_paths = sorted(
        (args.generation_run / "queries").glob("*.json"),
        key=lambda path: int(path.stem),
    )
    for result_path in result_paths:
        pair = json.loads(result_path.read_text())
        query = queries[pair["query_id"]]
        judgments = judgments_by_query.get(pair["query_id"], [])
        needed_page_ids = {
            judgment["target_id"] for judgment in judgments if judgment["relevance"] >= 1
        }
        for arm_name in ("retrieved", "oracle"):
            needed_page_ids.update(
                citation["page_id"] for citation in pair[arm_name]["answer"]["citations"]
            )

        for page_id in needed_page_ids:
            if page_id in exported_pages:
                continue
            source = pages[page_id]
            page = source["page"]
            native_id = page["metadata"]["native_corpus_id"]
            image_name = f"{native_id:06d}.webp"
            export_page(args.materialization / source["image_path"], output_images / image_name)
            exported_pages[page_id] = {
                "pageId": page_id,
                "pageNumber": page["page_number"],
                "documentTitle": document_title(page["document_id"]),
                "width": page["width"],
                "height": page["height"],
                "imageUrl": f"/evidence/{image_name}",
            }

        gold_evidence = []
        for judgment in judgments:
            if judgment["relevance"] < 1:
                continue
            gold_evidence.append(
                {
                    "pageId": judgment["target_id"],
                    "relevance": judgment["relevance"],
                    "contentTypes": judgment["content_types"],
                    "boundingBoxes": judgment["bounding_boxes"],
                }
            )
        evidence_types = sorted(
            {
                content_type
                for judgment in judgments
                if judgment["relevance"] >= 1
                for content_type in judgment["content_types"]
            },
            key=lambda value: ("Text", "Table", "Chart", "Infographic").index(value)
            if value in ("Text", "Table", "Chart", "Infographic")
            else 99,
        )

        exported_queries.append(
            {
                "queryId": pair["query_id"],
                "nativeQueryId": pair["native_query_id"],
                "question": query["text"],
                "referenceAnswers": query["reference_answers"],
                "evidenceTypes": evidence_types,
                "queryTypes": query["metadata"]["query_types"],
                "goldEvidence": gold_evidence,
                "retrieved": public_arm(pair["retrieved"]),
                "oracle": public_arm(pair["oracle"]),
            }
        )

    demo = {
        "schemaVersion": "1",
        "dataset": "ViDoRe V3 HR",
        "runFingerprint": json.loads((args.generation_run / "manifest.json").read_text())[
            "fingerprint"
        ],
        "recordingNotice": (
            "Recorded evaluation replay. Every answer, source page, citation, token count, "
            "cost, and latency measurement comes from the completed Gate C run."
        ),
        "queries": exported_queries,
        "pages": exported_pages,
        "summaries": {
            "retrieval": [
                {
                    "label": label,
                    "source": source,
                    **json.loads((ROOT / result_path).read_text())["aggregate"],
                }
                for label, source, result_path in (
                    ("BM25", "Shipped text", "results/bm25_fixed_markdown.json"),
                    ("MiniLM", "Shipped text", "results/dense_fixed_markdown.json"),
                    ("Hybrid RRF", "Shipped text", "results/hybrid_fixed_markdown.json"),
                    ("BM25", "Tesseract OCR", "results/bm25_fixed_ocr.json"),
                )
            ],
            "generation": json.loads(
                (ROOT / "examples/gate_c_generation_summary.json").read_text()
            ),
            "evidence": json.loads((ROOT / "examples/gate_c_evidence_breakdown.json").read_text()),
            "policyComparison": json.loads(
                (ROOT / "examples/gate_c_fixed_vs_structure_generation.json").read_text()
            ),
        },
    }
    data_path = args.frontend / "src/data/demo.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(demo, indent=2) + "\n")
    print(
        f"Exported {len(exported_queries)} real queries and {len(exported_pages)} evidence pages"
    )


if __name__ == "__main__":
    main()
