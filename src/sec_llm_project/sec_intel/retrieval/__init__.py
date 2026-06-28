"""Hybrid retrieval: BM25 + dense + metadata filter + fusion + rerank (plan item 3)."""

from __future__ import annotations

from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .hybrid import HybridRetriever

__all__ = ["BM25Index", "HybridRetriever", "reciprocal_rank_fusion", "weighted_fusion"]
