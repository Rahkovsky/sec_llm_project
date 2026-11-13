# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Local LLM inference via Ollama.

The provider lazy-uses only the standard library (HTTP to the Ollama server), so
the package imports cleanly with no third-party SDK. It implements the
:class:`LLMProvider` contract, so the rest of the platform is agnostic to the
backend (plan item 1).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import GenerationResult, Message


class OllamaLLM:
    """Local inference via an Ollama server (default http://localhost:11434)."""

    def __init__(self, model: str, base_url: str = "", temperature: float = 0.0,
                 max_tokens: int = 768, seed: int = 7, timeout: float = 120.0,
                 num_ctx: int = 8192, *, think: bool = False,
                 presence_penalty: float = 0.0) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.think = think
        self.presence_penalty = presence_penalty

    def generate(self, messages: list[Message], *, temperature: float | None = None,
                 max_tokens: int | None = None, json_schema: dict[str, Any] | None = None
                 ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            # Thinking tokens would otherwise eat num_predict (see LLMConfig.think).
            "think": self.think,
            "options": {
                "presence_penalty": self.presence_penalty,
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self.num_ctx,
                "seed": self.seed,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        text = (body.get("message", {}) or {}).get("content", "")
        return GenerationResult(text=text, model=self.model, raw=body)
