"""Dataset adapters and registry."""

from vidore_rag.datasets.registry import DatasetRegistry
from vidore_rag.datasets.vidore_v3 import ViDoReV3Adapter

__all__ = ["DatasetRegistry", "ViDoReV3Adapter"]
