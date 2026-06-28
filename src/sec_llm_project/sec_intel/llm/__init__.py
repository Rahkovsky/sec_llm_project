"""Pluggable LLM abstraction (plan item 1).

Backends (mock/ollama/openai/anthropic) are selected through configuration via
:func:`build_llm`. Retrieval is kept separate from generation; an LLM provider
only knows how to turn messages into text.
"""

from __future__ import annotations

from .base import GenerationResult, LLMProvider, Message
from .factory import build_llm

__all__ = ["GenerationResult", "LLMProvider", "Message", "build_llm"]
