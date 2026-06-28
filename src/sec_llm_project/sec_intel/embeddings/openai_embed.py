# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""OpenAI embeddings backend (default production embedder).

Uses ``text-embedding-3-large`` by default. The ``openai`` SDK is imported
lazily so the package imports without it. Works against Azure OpenAI / gateways
via ``base_url`` for FedRAMP-compatible deployments.
"""

from __future__ import annotations

import os

from .base import EmbeddingInfo

# Native output dimensionality of common OpenAI embedding models.
_MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large",
                 api_key_env: str = "OPENAI_API_KEY", base_url: str = "",
                 normalize: bool = True, version: str = "v1") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.normalize = normalize
        self.info = EmbeddingInfo(
            backend="openai", model=model, dim=_MODEL_DIMS.get(model, 3072),
            normalize=normalize, version=version,
        )

    def _client(self):
        from openai import OpenAI  # lazy

        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"OpenAI embeddings need an API key in env var '{self.api_key_env}'."
            )
        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client().embeddings.create(model=self.model, input=texts)
        return [list(map(float, d.embedding)) for d in resp.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
