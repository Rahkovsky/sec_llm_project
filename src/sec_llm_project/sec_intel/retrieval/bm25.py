"""Pure-Python BM25 lexical retrieval (plan item 3).

A self-contained Okapi BM25 implementation so lexical retrieval works with no
third-party dependency. It complements dense retrieval by catching exact-term
matches (tickers, statute names, defined terms) that embeddings often blur.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from ..core.types import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Bumped when the persisted BM25 format changes, so a stale file is ignored.
_BM25_FORMAT = 1


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = [c.chunk_id for c in chunks]
        self.chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._docs: list[list[str]] = [tokenize(c.text) for c in chunks]
        self._tf: list[Counter[str]] = [Counter(doc) for doc in self._docs]
        self._doc_len: list[int] = [len(doc) for doc in self._docs]
        self._avgdl: float = (sum(self._doc_len) / len(self._docs)) if self._docs else 0.0
        self._df: Counter[str] = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        self._n = len(self._docs)
        self._idf: dict[str, float] = {
            term: math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def search(self, query: str, k: int,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if self._n == 0:
            return []
        q_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, cid in enumerate(self.chunk_ids):
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            tf = self._tf[i]
            dl = self._doc_len[i]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * (f * (self.k1 + 1)) / (denom or 1.0)
            if score > 0:
                scores.append((cid, score))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:k]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        """Persist the precomputed scoring tables (the expensive tokenization
        work) so a query session can load instead of rebuilding."""
        data = {
            "format": _BM25_FORMAT,
            "k1": self.k1,
            "b": self.b,
            "n": self._n,
            "avgdl": self._avgdl,
            "chunk_ids": self.chunk_ids,
            "doc_len": self._doc_len,
            "idf": self._idf,
            "tf": [dict(tf) for tf in self._tf],  # Counters -> plain dicts
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def from_saved(cls, path: str | Path) -> BM25Index | None:
        """Load a persisted index, bypassing __init__ (no re-tokenization).
        Returns None if the file is missing or a stale format."""
        p = Path(path)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("format") != _BM25_FORMAT:
            return None
        obj = cls.__new__(cls)
        obj.k1 = data["k1"]
        obj.b = data["b"]
        obj._n = data["n"]
        obj._avgdl = data["avgdl"]
        obj.chunk_ids = data["chunk_ids"]
        obj._doc_len = data["doc_len"]
        obj._idf = data["idf"]
        obj._tf = [Counter(tf) for tf in data["tf"]]
        obj.chunks = {}  # not used by search; texts live in the chunk store
        obj._docs = []
        obj._df = Counter()
        return obj
