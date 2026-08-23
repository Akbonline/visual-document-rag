"""Cache and verify the exact dense encoder required by the deployment bundle."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from vidore_rag.artifacts import read_json
from vidore_rag.retrieval.dense import SentenceTransformerEncoder


class DeploymentModel(BaseModel):
    model_name: str
    model_revision: str


def main() -> None:
    artifact_dir = Path(os.getenv("VIDORE_RAG_ARTIFACT_DIR", "deployment/artifacts"))
    model = DeploymentModel.model_validate(read_json(artifact_dir / "manifest.json"))
    expected_dimensions = int(
        np.load(artifact_dir / "embeddings.npy", mmap_mode="r").shape[1]
    )
    encoder = SentenceTransformerEncoder(
        model_name=model.model_name,
        revision=model.model_revision,
        device="cpu",
        local_files_only=False,
    )
    vector = encoder.encode(["deployment model verification"], batch_size=1)
    if vector.shape != (1, expected_dimensions):
        raise RuntimeError(
            "dense encoder dimensions do not match the committed corpus embeddings"
        )
    print(
        f"Cached {model.model_name}@{model.model_revision} "
        f"with {expected_dimensions} dimensions."
    )


if __name__ == "__main__":
    main()
