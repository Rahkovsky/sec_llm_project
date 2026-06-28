"""Dependency injection for LLM providers (plan items 1, 11)."""

from __future__ import annotations

from ..core.config import LLMConfig
from .base import LLMProvider


def build_llm(config: LLMConfig) -> LLMProvider:
    """Construct an :class:`LLMProvider` from configuration.

    The provider is chosen purely from ``config.provider``; no caller code
    changes when swapping local Ollama for a hosted API.
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
        )
    if provider in {"openai", "openai-compat", "openai_compatible"}:
        from .providers import OpenAICompatLLM

        return OpenAICompatLLM(
            model=config.model, base_url=config.base_url,
            api_key_env=config.api_key_env or "OPENAI_API_KEY",
            temperature=config.temperature, max_tokens=config.max_tokens,
            seed=config.seed, timeout=config.timeout,
        )
    if provider == "anthropic":
        from .providers import AnthropicLLM

        return AnthropicLLM(
            model=config.model, api_key_env=config.api_key_env or "ANTHROPIC_API_KEY",
            temperature=config.temperature, max_tokens=config.max_tokens,
            timeout=config.timeout, base_url=config.base_url,
        )
    raise ValueError(f"Unknown LLM provider '{config.provider}'")
