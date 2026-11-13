# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Local LLM inference via Ollama.

The provider lazy-uses only the standard library (HTTP to the Ollama server), so
the package imports cleanly with no third-party SDK. It implements the
:class:`LLMProvider` contract, so the rest of the platform is agnostic to the
backend (plan item 1).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from .base import GenerationResult, Message

# Per-call timing log (model, calling stage, Ollama's load/prompt/eval durations),
# one list per thread. tools/provenance_dag.py resets it per question and writes
# <qid>/timings.json.
_tls = threading.local()


def call_log() -> list[dict[str, Any]]:
    if not hasattr(_tls, "calls"):
        _tls.calls = []
    return _tls.calls


def reset_call_log() -> None:
    _tls.calls = []


class ModelScheduler:
    """Order concurrent requests so the loaded model is swapped as rarely as possible.

    On a machine that holds one model at a time (OLLAMA_MAX_LOADED_MODELS=1), a
    question alternates generator and judge several times, and each switch reloads
    ~8 GB. With several questions in flight (one thread each), this serves every
    waiting request for the loaded model first and switches only when every live
    worker is blocked on another model -- then to the model most of them wait for.
    Each answer goes through exactly the same calls; only their order across
    questions changes. One request runs at a time (Ollama serves them serially).
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._waiting: Counter[str] = Counter()
        self._active = 0
        self._busy = False
        self.current: str | None = None
        self.switches = 0

    @contextmanager
    def worker(self) -> Iterator[None]:
        with self._cv:
            self._active += 1
        try:
            yield
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def _may_run(self, model: str) -> bool:
        if self._busy:
            return False
        if model == self.current:
            return True
        if self._waiting[self.current or ""] or sum(self._waiting.values()) < self._active:
            return False  # the loaded model still has callers, or a worker may need it soon
        best = max((m for m, n in self._waiting.items() if n), key=lambda m: self._waiting[m])
        return model == best

    @contextmanager
    def turn(self, model: str) -> Iterator[None]:
        with self._cv:
            self._waiting[model] += 1
            self._cv.notify_all()
            while not self._may_run(model):
                self._cv.wait()
            self._waiting[model] -= 1
            if model != self.current:
                self.current = model
                self.switches += 1
            self._busy = True
        try:
            yield
        finally:
            with self._cv:
                self._busy = False
                self._cv.notify_all()


# Set by a driver that runs questions concurrently; None = plain serial calls.
SCHEDULER: ModelScheduler | None = None


def _stage() -> str:
    """Name of the first caller outside this package's llm module (e.g. verify)."""
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename.endswith(("providers.py", "base.py")):
        f = f.f_back
    return f.f_code.co_name if f is not None else "?"


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
        t0 = time.perf_counter()
        with SCHEDULER.turn(self.model) if SCHEDULER is not None else nullcontext():
            t_run = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as exc:  # pragma: no cover - network
                raise RuntimeError(f"Ollama request failed: {exc}") from exc
        ns = 1e-9
        call_log().append({
            "model": self.model, "stage": _stage(),
            "queued_s": round(t_run - t0, 3),
            "wall_s": round(time.perf_counter() - t_run, 3),
            "load_s": round(body.get("load_duration", 0) * ns, 3),
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "prompt_s": round(body.get("prompt_eval_duration", 0) * ns, 3),
            "out_tokens": body.get("eval_count", 0),
            "eval_s": round(body.get("eval_duration", 0) * ns, 3),
        })
        text = (body.get("message", {}) or {}).get("content", "")
        return GenerationResult(text=text, model=self.model, raw=body)
