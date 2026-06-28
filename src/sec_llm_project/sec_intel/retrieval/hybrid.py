"""Hybrid retriever orchestration (plan item 3).

Combines lexical (BM25) and dense retrieval with optional metadata filtering,
reciprocal-rank / weighted fusion, and optional cross-encoder reranking. Each
returned result exposes its per-retriever score components for transparency
(plan item 10).
"""

from __future__ import annotations

import time
from typing import Any

from ..core.config import RetrievalConfig
from ..core.types import Chunk, RetrievalResult
from ..embeddings.base import Embedder
from ..index.store import SECIndex
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion, weighted_fusion


def _matches(chunk: Chunk, where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    meta = chunk.metadata.to_dict()
    meta["section_title"] = chunk.section_title
    meta["item_number"] = chunk.item_number
    for key, want in where.items():
        have = meta.get(key)
        if isinstance(want, list | tuple | set):
            if have not in {str(w) for w in want}:
                return False
        elif str(have) != str(want):
            return False
    return True


class HybridRetriever:
    def __init__(self, index: SECIndex, embedder: Embedder,
                 config: RetrievalConfig | None = None, reranker: Any = None) -> None:
        self.index = index
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._reranker = reranker
        self._bm25: BM25Index | None = None
        self.last_latency_ms: float = 0.0

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            # Prefer a BM25 index persisted at indexing time (no re-tokenization),
            # but only when the index was loaded from disk — then bm25.json is known
            # to match these chunks. A freshly built in-memory index (tests, ad-hoc
            # builds) must rebuild from its own chunks, not adopt a foreign file.
            persisted = None
            path = getattr(self.index, "path", "")
            if path and getattr(self.index, "loaded_from_disk", False):
                from pathlib import Path
                persisted = BM25Index.from_saved(Path(path, "bm25.json"))
            self._bm25 = persisted or BM25Index(self.index.all_chunks())
        return self._bm25

    def invalidate(self) -> None:
        """Drop cached lexical index after the underlying index changes."""
        self._bm25 = None

    def retrieve(self, query: str, *, top_k: int | None = None,
                 filters: dict[str, Any] | None = None) -> list[RetrievalResult]:
        cfg = self.config
        top_k = top_k or cfg.top_k
        started = time.perf_counter()

        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        if cfg.use_dense:
            qvec = self.embedder.embed_query(query)
            dense_scores = dict(self.index.dense_search(qvec, cfg.candidate_k, filters))

        if cfg.use_bm25:
            allowed = None
            if filters:
                allowed = {
                    c.chunk_id for c in self.index.all_chunks() if _matches(c, filters)
                }
            bm25_scores = dict(self._bm25_index().search(query, cfg.candidate_k, allowed))

        fused = self._fuse(dense_scores, bm25_scores)
        results: list[RetrievalResult] = []
        for rank, (cid, score) in enumerate(
            sorted(fused.items(), key=lambda t: t[1], reverse=True)
        ):
            chunk = self.index.get(cid)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk=chunk, score=score, rank=rank,
                    components={
                        "dense": round(dense_scores.get(cid, 0.0), 6),
                        "bm25": round(bm25_scores.get(cid, 0.0), 6),
                        "fused": round(score, 6),
                    },
                )
            )

        if self._reranker is not None and results:
            pool = results[: max(top_k * 3, cfg.candidate_k)]
            results = self._reranker.rerank(query, pool, top_k=top_k)
        else:
            results = results[:top_k]

        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return results

    def _fuse(self, dense: dict[str, float], bm25: dict[str, float]) -> dict[str, float]:
        cfg = self.config
        if not dense:
            return bm25
        if not bm25:
            return dense
        if cfg.fusion == "weighted":
            return weighted_fusion([dense, bm25], [cfg.dense_weight, cfg.bm25_weight])
        dense_ranked = [cid for cid, _ in sorted(dense.items(), key=lambda t: t[1], reverse=True)]
        bm25_ranked = [cid for cid, _ in sorted(bm25.items(), key=lambda t: t[1], reverse=True)]
        return reciprocal_rank_fusion(
            [dense_ranked, bm25_ranked], k=cfg.rrf_k,
            weights=[cfg.dense_weight, cfg.bm25_weight],
        )
