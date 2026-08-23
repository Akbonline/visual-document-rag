from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vidore_rag.generation.models import GenerationConfig


def load_generation_config(path: Path) -> GenerationConfig:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return GenerationConfig.model_validate(raw)
