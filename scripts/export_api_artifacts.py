from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from vidore_rag.artifacts import read_json, read_jsonl, sha256_file
from vidore_rag.ingestion.materialize import MaterializationManifest
from vidore_rag.retrieval.dense import DenseIndexManifest
from vidore_rag.retrieval.indexing import TextIndexManifest, resolve_projection

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZATION = (
    ROOT
    / ".artifacts/vidore_v3_hr/materialized"
    / "c30f10d4d97b89e4d04d762683aabaa18de86731ef2bbbc7f1d75f87f4a2ebe3"
)
TEXT_INDEX = (
    ROOT
    / ".artifacts/vidore_v3_hr/text_indexes"
    / "bf971d36baf939425ff8062e1ef1c25e6aa13ad8188c23e37f495b26807134b1"
)
DENSE_INDEX = (
    ROOT
    / ".artifacts/vidore_v3_hr/dense_indexes"
    / "ae9a5faa4ec9ced01608205996fcff7bd24e9e5942e7e61b104e5b2bd40b2a14"
)
OUTPUT = ROOT / "deployment/artifacts"


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def write_public_pages(source: Path, destination: Path) -> int:
    rows = read_jsonl(source)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            page = row["page"]
            page["image_uri"] = None
            handle.write(json.dumps(page, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, destination)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deployable API retrieval artifacts")
    parser.add_argument("--materialization", type=Path, default=MATERIALIZATION)
    parser.add_argument("--text-index", type=Path, default=TEXT_INDEX)
    parser.add_argument("--dense-index", type=Path, default=DENSE_INDEX)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    materialized = MaterializationManifest.model_validate(
        read_json(args.materialization / "manifest.json")
    )
    text_index = TextIndexManifest.model_validate(read_json(args.text_index / "manifest.json"))
    dense_index = DenseIndexManifest.model_validate(read_json(args.dense_index / "manifest.json"))
    if dense_index.text_index_fingerprint != text_index.fingerprint:
        raise ValueError("dense and text artifacts do not share a fingerprint")

    sources = {
        "chunks.jsonl": args.text_index / text_index.chunks_path,
        "embeddings.npy": args.dense_index / dense_index.embeddings_path,
        "queries.jsonl": args.materialization / materialized.queries_path,
        "judgments.jsonl": args.materialization / materialized.judgments_path,
    }
    for name, source in sources.items():
        atomic_copy(source, args.output / name)
    page_count = write_public_pages(
        args.materialization / materialized.pages_path,
        args.output / "pages.jsonl",
    )

    files = {
        name: sha256_file(args.output / name)
        for name in (*sources, "pages.jsonl")
    }
    manifest = {
        "schema_version": "1",
        "dataset_id": materialized.dataset_id,
        "dataset_fingerprint": materialized.fingerprint,
        "text_index_fingerprint": text_index.fingerprint,
        "dense_index_fingerprint": dense_index.fingerprint,
        "projection_fingerprint": resolve_projection(text_index).projection_fingerprint,
        "model_name": dense_index.model_name,
        "model_revision": dense_index.model_revision,
        "chunk_count": text_index.chunk_count,
        "page_count": page_count,
        "query_count": materialized.query_count,
        "files": files,
    }
    temporary_manifest = args.output / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, args.output / "manifest.json")
    print(
        f"Exported {text_index.chunk_count} chunks, {page_count} pages, "
        f"and {materialized.query_count} queries to {args.output}"
    )


if __name__ == "__main__":
    main()
