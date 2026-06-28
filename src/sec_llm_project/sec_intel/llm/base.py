"""LLM provider interface shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str


@dataclass
class GenerationResult:
    text: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal generation contract.

    Implementations must be deterministic when ``temperature == 0`` so prompts
    are reproducible (plan item 10).
    """

    name: str

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        ...
