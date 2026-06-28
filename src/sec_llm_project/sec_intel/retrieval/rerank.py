# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Optional cross-encoder reranking (plan item 3).

Reranking re-scores a small candidate set with a query-document cross-encoder,
which is far more precise than bi-encoder cosine for the final ordering. The
``sentence-transformers`` CrossEncoder is imported lazily.
"""

from __future__ import annotations

from ..core.types import RetrievalResult


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # lazy

        self.model_name = model
        self._model = CrossEncoder(model)

    def rerank(self, query: str, results: list[RetrievalResult],
               top_k: int | None = None) -> list[RetrievalResult]:
        if not results:
            return results
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)
        for r, s in zip(results, scores, strict=False):
            r.components["rerank"] = float(s)
            r.score = float(s)
        results.sort(key=lambda r: r.score, reverse=True)
        for rank, r in enumerate(results):
            r.rank = rank
        return results[:top_k] if top_k else results
