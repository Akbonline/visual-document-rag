"""Dataset adapters and registry."""

from vidore_rag.datasets.base import DatasetAdapter, DatasetDescriptor, PageAsset
from vidore_rag.datasets.defaults import build_default_registry
from vidore_rag.datasets.registry import DatasetRegistry
from vidore_rag.datasets.vidore_v3 import ViDoReV3Adapter

__all__ = [
    "DatasetAdapter",
    "DatasetDescriptor",
    "DatasetRegistry",
    "PageAsset",
    "ViDoReV3Adapter",
    "build_default_registry",
]
