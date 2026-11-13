"""Hybrid retriever orchestration (plan item 3).

Combines lexical (BM25) and dense retrieval with optional metadata filtering,
reciprocal-rank / weighted fusion, and optional cross-encoder reranking. Each
returned result exposes its per-retriever score components for transparency
(plan item 10).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..core.config import RetrievalConfig
from ..core.types import Chunk, RetrievalResult
from ..embeddings.base import Embedder
from ..index.store import SECIndex
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .period import period_filter


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


def _mean_unit(vecs: list[list[float]]) -> list[float]:
    """Mean of the L2-normalized vectors (the original vector when there is one)."""
    if len(vecs) == 1:
        return vecs[0]
    import numpy as np
    a = np.asarray(vecs, dtype=np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return a.mean(axis=0).tolist()


class HybridRetriever:
    # Per-call outputs read back by the caller after retrieve(). Thread-local, so
    # concurrent questions (tools/provenance_dag.py --workers) do not see each other's.
    @property
    def last_period(self) -> list[str] | None:
        return getattr(self._tls, "period", None)

    @last_period.setter
    def last_period(self, v: list[str] | None) -> None:
        self._tls.period = v

    @property
    def last_expansions(self) -> list[str]:
        return getattr(self._tls, "expansions", [])

    @last_expansions.setter
    def last_expansions(self, v: list[str]) -> None:
        self._tls.expansions = v

    @property
    def last_latency_ms(self) -> float:
        return getattr(self._tls, "latency_ms", 0.0)

    @last_latency_ms.setter
    def last_latency_ms(self, v: float) -> None:
        self._tls.latency_ms = v

    def __init__(self, index: SECIndex, embedder: Embedder,
                 config: RetrievalConfig | None = None, reranker: Any = None,
                 expander: Any = None) -> None:
        self.index = index
        # Optional query expander (retrieval/rewrite.py): expand(query, filters) -> [texts]
        self.expander = expander
        self._tls = threading.local()
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._reranker = reranker
        self._bm25: BM25Index | None = None
        # filter -> allowed chunk ids. Building one scans every chunk (~660k on the
        # full index, seconds per query); repeated tickers reuse the set.
        self._allowed_cache: dict[str, frozenset[str]] = {}
        self._years_by_ticker: dict[str, set[str]] | None = None

    def _period_years(self, query: str, filters: dict[str, Any] | None) -> list[str] | None:
        """Fiscal years of the period named in a question about one ticker, or None."""
        if not (self.config.period_filter and filters and "ticker" in filters
                and "fiscal_year" not in filters
                and not isinstance(filters["ticker"], list | tuple | set)):
            return None
        if self._years_by_ticker is None:
            years: dict[str, set[str]] = {}
            for c in self.index.catalog():
                years.setdefault(str(c.metadata.ticker), set()).add(str(c.metadata.fiscal_year))
            self._years_by_ticker = years
        return period_filter(query, self._years_by_ticker.get(str(filters["ticker"]), ()))

    def _allowed_ids(self, filters: dict[str, Any]) -> frozenset[str]:
        key = repr(sorted((k, sorted(map(str, v)) if isinstance(v, list | tuple | set)
                           else str(v)) for k, v in filters.items()))
        allowed = self._allowed_cache.get(key)
        if allowed is None and list(filters) == ["ticker"] and not isinstance(
                filters["ticker"], list | tuple | set):
            # Common case: one ticker. Bucket every chunk by ticker in a single pass
            # (same str() comparison as _matches) so each new ticker is a lookup.
            if not self._allowed_cache.get("__by_ticker_built__"):
                buckets: dict[str, set[str]] = {}
                for c in self.index.catalog():
                    buckets.setdefault(str(c.metadata.ticker), set()).add(c.chunk_id)
                for t, ids in buckets.items():
                    self._allowed_cache[repr([("ticker", t)])] = frozenset(ids)
                self._allowed_cache["__by_ticker_built__"] = frozenset({"1"})
            allowed = self._allowed_cache.get(key, frozenset())
        if allowed is None:
            allowed = frozenset(
                c.chunk_id for c in self.index.catalog() if _matches(c, filters))
            self._allowed_cache[key] = allowed
        return allowed

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
        self._allowed_cache.clear()

    def retrieve(self, query: str, *, top_k: int | None = None,
                 filters: dict[str, Any] | None = None) -> list[RetrievalResult]:
        cfg = self.config
        top_k = top_k or cfg.top_k
        started = time.perf_counter()
        period = self.last_period = self._period_years(query, filters)
        soft = period is not None and cfg.period_mode == "soft"
        if period is not None and not soft:
            filters = {**filters, "fiscal_year": period}
        boost = cfg.period_boost if soft else 0.0

        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        qvecs: list[list[float]] = []
        self.last_expansions = []
        if cfg.use_dense:
            qvecs = [self.embedder.embed_query(query)]
            if self.expander is not None:
                self.last_expansions = self.expander.expand(query, filters)
                qvecs += [self.embedder.embed_query(t) for t in self.last_expansions]
            # mean cosine over the query set == cosine-order of the mean unit vector
            qvec = _mean_unit(qvecs)
            dense_scores = dict(self.index.dense_search(qvec, cfg.candidate_k, filters))
            if soft:
                # in-period candidates may rank below the unfiltered top candidate_k
                dense_scores.update(self.index.dense_search(
                    qvec, cfg.candidate_k, {**filters, "fiscal_year": period}))

        if cfg.use_bm25:
            allowed = self._allowed_ids(filters) if filters else None
            bm25_scores = dict(self._bm25_index().search(query, cfg.candidate_k, allowed))

        fused = self._fuse(dense_scores, bm25_scores)
        results: list[RetrievalResult] = []
        for rank, (cid, score) in enumerate(
            sorted(fused.items(), key=lambda t: t[1], reverse=True)
        ):
            chunk = self.index.get(cid)
            if chunk is None:
                continue
            if boost and str(chunk.metadata.fiscal_year) in period:
                score += boost
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

        if boost:  # the cosine scale is the dense one; with BM25 fusion on, the boost is cruder
            results.sort(key=lambda r: r.score, reverse=True)
            for rank, r in enumerate(results):
                r.rank = rank
        if cfg.mmr_lambda is not None and results and cfg.use_dense:
            results = self._mmr(qvecs, results[: max(cfg.mmr_pool, top_k)], top_k,
                                cfg.mmr_lambda, boost=boost, period=period)
        if self._reranker is not None and results:
            pool = results[: max(top_k * 3, cfg.candidate_k)]
            results = self._reranker.rerank(query, pool, top_k=top_k)
        else:
            results = results[:top_k]

        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        return results

    def _mmr(self, qvecs: list[list[float]], pool: list[RetrievalResult], top_k: int,
             lam: float, boost: float = 0.0, period: list[str] | None = None
             ) -> list[RetrievalResult]:
        """Maximal marginal relevance: trade query relevance against redundancy.

        Relevance is cos(query, chunk) from the stored vectors -- the mean over
        the query set when the query was expanded -- also with BM25 fusion on, so
        lambda keeps the scale it was tuned at; redundancy is the max cosine to the
        chunks already picked. Form/section agnostic.
        """
        import numpy as np

        vecs = self.index.vectors([r.chunk.chunk_id for r in pool])
        pool = [r for r in pool if r.chunk.chunk_id in vecs]
        if len(pool) <= 1:
            return pool[:top_k]
        emb = np.asarray([vecs[r.chunk.chunk_id] for r in pool], dtype=np.float32)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        q = np.asarray(qvecs, dtype=np.float32)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        rel = (emb @ q.T).mean(axis=1)
        if boost and period:  # soft period filter: same boost as the pool score
            rel = rel + boost * np.asarray(
                [str(r.chunk.metadata.fiscal_year) in period for r in pool], dtype=np.float32)
        chosen: list[int] = []
        rest = list(range(len(pool)))
        while rest and len(chosen) < top_k:
            val = rel[rest]
            if chosen:
                val = lam * val - (1 - lam) * (emb[rest] @ emb[chosen].T).max(axis=1)
            j = rest[int(np.argmax(val))]
            chosen.append(j)
            rest.remove(j)
        out = []
        for rank, j in enumerate(chosen):
            r = pool[j]
            out.append(RetrievalResult(chunk=r.chunk, score=r.score, rank=rank,
                                       components={**r.components, "mmr_rank": rank}))
        return out

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
