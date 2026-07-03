"""Lightweight span tracing for QC and observability.

Same philosophy as :class:`AuditLogger`: the core is dependency-free and defaults
to a no-op, so tracing never adds weight unless turned on. Three backends:

* :class:`NullTracer`  — default; zero cost, zero deps.
* :class:`JsonlTracer` — nested spans to a JSONL file (still zero deps). Good for
  local grep/jq inspection and as an always-on audit of each request's stages.
* :class:`OTelTracer`  — optional (``pip install '.[observability]'``). Exports
  OpenTelemetry spans (with OpenInference attributes) to a collector such as a
  self-hosted **Arize Phoenix** at ``http://localhost:6006/v1/traces`` — traces,
  retrieval inspection, and evals, all local and free. See docs/OBSERVABILITY.md.

A span has a single method, :meth:`Span.set`, to attach attributes as a stage
runs. A few keys are mapped to OpenInference conventions so Phoenix renders them
richly: ``input``/``output`` (chain & LLM I/O), ``model`` (LLM name), and
``documents`` (a list of ``{id, score, content}`` for retriever spans).
"""
from __future__ import annotations

import atexit
import contextvars
import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

# OpenInference span kinds (Phoenix renders these specially). Plain strings so no
# dependency is needed just to label a span.
CHAIN, RETRIEVER, LLM, RERANKER, EVALUATOR = (
    "CHAIN", "RETRIEVER", "LLM", "RERANKER", "EVALUATOR")

_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sec_intel_span", default=None)
_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sec_intel_trace", default=None)


@runtime_checkable
class Span(Protocol):
    def set(self, **attrs: Any) -> None: ...


class Tracer(Protocol):
    def span(self, name: str, *, kind: str = CHAIN, **attrs: Any) -> Any:
        """Context manager yielding a :class:`Span`."""
        ...


class _NullSpan:
    def set(self, **attrs: Any) -> None:
        return None


class NullTracer:
    """Default tracer: does nothing, allocates nothing meaningful."""

    @contextmanager
    def span(self, name: str, *, kind: str = CHAIN, **attrs: Any) -> Iterator[Span]:
        yield _NullSpan()


@dataclass
class _DictSpan:
    attrs: dict[str, Any] = field(default_factory=dict)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


@dataclass
class JsonlTracer:
    """Nested spans appended to a JSONL file. No dependencies.

    Each line is one completed span with ``trace_id``/``span_id``/``parent_id`` so
    a request's stage tree can be reconstructed (and replayed into Phoenix later).
    """

    path: str = "logs/traces.jsonl"

    def _write(self, record: dict[str, Any]) -> None:
        parent = os.path.dirname(self.path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def span(self, name: str, *, kind: str = CHAIN, **attrs: Any) -> Iterator[Span]:
        span_id = uuid.uuid4().hex[:16]
        parent = _current_span.get()
        is_root = parent is None
        trace_id = uuid.uuid4().hex if is_root else (_current_trace.get() or uuid.uuid4().hex)
        t_tok = _current_trace.set(trace_id) if is_root else None
        s_tok = _current_span.set(span_id)
        span = _DictSpan(dict(attrs))
        start = time.perf_counter()
        error: str | None = None
        try:
            yield span
        except Exception as exc:  # record failures in the trace, then re-raise
            error = repr(exc)
            raise
        finally:
            _current_span.reset(s_tok)
            if t_tok is not None:
                _current_trace.reset(t_tok)
            record = {
                "trace_id": trace_id, "span_id": span_id, "parent_id": parent,
                "name": name, "kind": kind,
                "ms": round((time.perf_counter() - start) * 1000, 2),
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "attributes": span.attrs,
            }
            if error is not None:
                record["error"] = error
            self._write(record)


class _OTelSpan:
    """Adapts :meth:`Span.set` onto an OpenTelemetry span with OpenInference keys."""

    _RENAME: ClassVar[dict[str, str]] = {
        "input": "input.value", "output": "output.value", "model": "llm.model_name"}

    def __init__(self, otel_span: Any) -> None:
        self._s = otel_span

    def set(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            if value is None:
                continue
            if key == "documents":
                for i, doc in enumerate(value):
                    base = f"retrieval.documents.{i}.document"
                    self._s.set_attribute(f"{base}.id", str(doc.get("id", "")))
                    self._s.set_attribute(f"{base}.content", str(doc.get("content", "")))
                    if doc.get("score") is not None:
                        self._s.set_attribute(f"{base}.score", float(doc["score"]))
                continue
            name = self._RENAME.get(key, key)
            self._s.set_attribute(
                name, value if isinstance(value, str | bool | int | float)
                else json.dumps(value, default=str))


class OTelTracer:
    """OpenTelemetry backend. Optional deps; exports to an OTLP collector (Phoenix)."""

    def __init__(self, endpoint: str, service: str = "sec-intel") -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The 'phoenix' tracing backend needs the observability extra: "
                'uv pip install -e ".[observability]"  (or set backend: jsonl).'
            ) from exc

        provider = TracerProvider(resource=Resource.create({"service.name": service}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        self._provider = provider
        self._tracer = provider.get_tracer(service)
        atexit.register(provider.shutdown)  # flush the batch on process exit (CLI)

    @contextmanager
    def span(self, name: str, *, kind: str = CHAIN, **attrs: Any) -> Iterator[Span]:
        with self._tracer.start_as_current_span(name) as otel_span:
            otel_span.set_attribute("openinference.span.kind", kind)
            wrapper = _OTelSpan(otel_span)
            wrapper.set(**attrs)
            yield wrapper


def build_tracer(config: Any) -> Tracer:
    """Construct a tracer from an ``ObservabilityConfig`` (or return a no-op)."""
    if not getattr(config, "enabled", False):
        return NullTracer()
    backend = str(getattr(config, "backend", "jsonl")).lower()
    if backend in {"otel", "phoenix"}:
        return OTelTracer(
            str(getattr(config, "endpoint", "http://localhost:6006/v1/traces")),
            service=str(getattr(config, "service", "sec-intel")))
    return JsonlTracer(str(getattr(config, "path", "logs/traces.jsonl")))
