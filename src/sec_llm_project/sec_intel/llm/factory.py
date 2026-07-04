"""Dependency injection for LLM providers (plan items 1, 11)."""

from __future__ import annotations

from ..core.config import LLMConfig
from .base import LLMProvider


def build_llm(config: LLMConfig) -> LLMProvider:
    """Construct an :class:`LLMProvider` from configuration.

    The provider is chosen purely from ``config.provider``: ``mock`` (offline,
    deterministic — tests/CI/demo) or ``ollama`` (local inference). No caller code
    changes when swapping between them.
    """
    provider = (config.provider or "mock").lower()
    if provider == "mock":
        from .mock import MockLLM

        return MockLLM(model=config.model)
    if provider == "ollama":
        from .providers import OllamaLLM

        return OllamaLLM(
            model=config.model, base_url=config.base_url, temperature=config.temperature,
            max_tokens=config.max_tokens, seed=config.seed, timeout=config.timeout,
            num_ctx=config.num_ctx,
        )
    raise ValueError(f"Unknown LLM provider '{config.provider}' (expected 'mock' or 'ollama')")
