# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Network-backed LLM providers: Ollama, OpenAI-compatible, Anthropic.

Each provider lazy-imports its SDK so the package imports cleanly with only the
standard library present. They share the :class:`LLMProvider` contract, so the
rest of the platform is agnostic to which one is configured (plan item 1).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import GenerationResult, Message


def _api_key(api_key_env: str) -> str:
    if not api_key_env:
        return ""
    key = os.environ.get(api_key_env, "")
    if not key:
        raise RuntimeError(
            f"Expected API key in environment variable '{api_key_env}' but it is unset."
        )
    return key


class OllamaLLM:
    """Local inference via an Ollama server (default http://localhost:11434)."""

    def __init__(self, model: str, base_url: str = "", temperature: float = 0.0,
                 max_tokens: int = 768, seed: int = 7, timeout: float = 120.0) -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout

    def generate(self, messages: list[Message], *, temperature: float | None = None,
                 max_tokens: int | None = None, json_schema: dict[str, Any] | None = None
                 ) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
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


class OpenAICompatLLM:
    """OpenAI-compatible chat completions (OpenAI, vLLM, LM Studio, gateways)."""

    def __init__(self, model: str, base_url: str = "", api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 768, seed: int = 7,
                 timeout: float = 120.0) -> None:
        self.name = f"openai:{model}"
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout

    def generate(self, messages: list[Message], *, temperature: float | None = None,
                 max_tokens: int | None = None, json_schema: dict[str, Any] | None = None
                 ) -> GenerationResult:
        from openai import OpenAI  # lazy

        client_kwargs: dict[str, Any] = {"api_key": _api_key(self.api_key_env)}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "seed": self.seed,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": json_schema, "strict": True},
            }
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            model=self.model,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            raw=resp,
        )


class AnthropicLLM:
    """Anthropic Messages API."""

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 768, timeout: float = 120.0,
                 base_url: str = "") -> None:
        self.name = f"anthropic:{model}"
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url

    def generate(self, messages: list[Message], *, temperature: float | None = None,
                 max_tokens: int | None = None, json_schema: dict[str, Any] | None = None
                 ) -> GenerationResult:
        from anthropic import Anthropic  # lazy

        client_kwargs: dict[str, Any] = {"api_key": _api_key(self.api_key_env)}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = Anthropic(**client_kwargs)

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in {"user", "assistant"}
        ]
        if json_schema is not None:
            # Anthropic has no strict json_schema param; instruct via system text.
            system = (system + "\n\nRespond ONLY with JSON matching this schema:\n"
                      + json.dumps(json_schema)).strip()
        resp = client.messages.create(
            model=self.model,
            system=system or None,
            messages=convo,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return GenerationResult(
            text=text, model=self.model, finish_reason=resp.stop_reason or "stop", raw=resp
        )
