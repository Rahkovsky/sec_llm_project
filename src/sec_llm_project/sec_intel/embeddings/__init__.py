"""Embedding abstraction with versioning metadata (plan item 2)."""

from __future__ import annotations

from .base import Embedder, EmbeddingInfo
from .factory import build_embedder

__all__ = ["Embedder", "EmbeddingInfo", "build_embedder"]
