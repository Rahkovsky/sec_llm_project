"""Embedder dependency injection (plan item 2).

Default production embeddings use OpenAI ``text-embedding-3-large``. Local/private
deployments use Nomic (``nomic-embed-text``, US-origin) via sentence-transformers.
The offline ``hashing`` backend needs no dependencies and powers tests/CI/demo.
"""

from __future__ import annotations

from ..core.config import EmbeddingConfig
from .base import Embedder


def build_embedder(config: EmbeddingConfig) -> Embedder:
    provider = (config.provider or "hashing").lower()
    if provider == "hashing":
        from .hashing import HashingEmbedder

        return HashingEmbedder(dim=config.dim, normalize=config.normalize, version=config.version)
    if provider == "openai":
        from .openai_embed import OpenAIEmbedder

        return OpenAIEmbedder(
            model=config.model or "text-embedding-3-large",
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            base_url=config.base_url, normalize=config.normalize, version=config.version,
        )
    if provider in {"nomic", "huggingface", "hf", "sentence-transformers"}:
        from .huggingface import HuggingFaceEmbedder

        model = config.model or "nomic-ai/nomic-embed-text-v1.5"
        return HuggingFaceEmbedder(
            model=model, normalize=config.normalize, version=config.version,
            revision=config.revision or None,
        )
    raise ValueError(f"Unknown embedding provider '{config.provider}'")
