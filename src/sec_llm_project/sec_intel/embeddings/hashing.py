"""Deterministic hashing embedder (offline default).

A feature-hashing bag-of-words model: no weights, no downloads, fully
deterministic. It is not as strong as a neural embedder, but it gives a real,
cosine-comparable dense vector so the dense-retrieval path is exercised in
tests and demos without any model dependency.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

from .base import EmbeddingInfo

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Iterable[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Feature-hashing embedder implementing the ``Embedder`` protocol."""

    def __init__(self, dim: int = 256, normalize: bool = True, version: str = "v1") -> None:
        self.dim = dim
        self.normalize = normalize
        self.info = EmbeddingInfo(
            backend="hashing", model=f"hashing-{dim}", dim=dim,
            normalize=normalize, version=version,
        )

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        if self.normalize:
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
