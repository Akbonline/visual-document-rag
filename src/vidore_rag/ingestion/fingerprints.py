from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stage_fingerprint(
    *,
    stage_name: str,
    implementation_version: str,
    config: Mapping[str, Any],
    upstream_fingerprints: Sequence[str] = (),
    model_revision: str | None = None,
) -> str:
    payload = {
        "config": config,
        "implementation_version": implementation_version,
        "model_revision": model_revision,
        "stage_name": stage_name,
        "upstream_fingerprints": list(upstream_fingerprints),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactFingerprint:
    stage_name: str
    implementation_version: str
    config: Mapping[str, Any]
    upstream_fingerprints: tuple[str, ...] = ()
    model_revision: str | None = None

    @property
    def digest(self) -> str:
        return stage_fingerprint(**asdict(self))

