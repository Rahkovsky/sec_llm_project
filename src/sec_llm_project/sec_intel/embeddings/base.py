"""Embedder interface and version descriptor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddingInfo:
    """Reproducibility metadata stored alongside the index (plan item 2).

    Querying an index with a mismatched embedder is a silent correctness bug;
    this descriptor lets the index detect and reject such mismatches.
    """

    backend: str
    model: str
    dim: int
    normalize: bool
    version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return f"{self.backend}/{self.model}@{self.version}/d{self.dim}"


@runtime_checkable
class Embedder(Protocol):
    info: EmbeddingInfo

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
