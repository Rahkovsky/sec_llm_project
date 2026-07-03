"""Zero-dependency tracer backends and their selection."""

from __future__ import annotations

import json

from sec_llm_project.sec_intel.core.config import ObservabilityConfig
from sec_llm_project.sec_intel.core.tracing import (
    JsonlTracer,
    NullTracer,
    build_tracer,
)


def test_null_tracer_is_noop_and_yields_a_span():
    tracer = NullTracer()
    with tracer.span("x", kind="CHAIN", input="q") as s:
        s.set(anything=1)  # must not raise


def test_jsonl_tracer_writes_nested_spans(tmp_path):
    path = tmp_path / "traces.jsonl"
    tracer = JsonlTracer(str(path))
    with tracer.span("answer", kind="CHAIN", input="q") as root:
        root.set(confidence=0.9)
        with tracer.span("retrieve", kind="RETRIEVER") as rs:
            rs.set(n_retrieved=3)

    lines = [json.loads(x) for x in path.read_text().splitlines()]
    # Children are written before parents (spans close inner-first).
    by_name = {r["name"]: r for r in lines}
    assert set(by_name) == {"answer", "retrieve"}
    root, child = by_name["answer"], by_name["retrieve"]
    # Same trace, and the child points at the root as parent.
    assert child["trace_id"] == root["trace_id"]
    assert child["parent_id"] == root["span_id"]
    assert root["parent_id"] is None
    assert root["attributes"]["confidence"] == 0.9
    assert child["attributes"]["n_retrieved"] == 3
    assert "ms" in root and "kind" in root


def test_jsonl_tracer_records_errors(tmp_path):
    path = tmp_path / "traces.jsonl"
    tracer = JsonlTracer(str(path))
    try:
        with tracer.span("boom"):
            raise ValueError("nope")
    except ValueError:
        pass
    rec = json.loads(path.read_text().splitlines()[0])
    assert "error" in rec and "nope" in rec["error"]


def test_build_tracer_selects_backend(tmp_path):
    assert isinstance(build_tracer(ObservabilityConfig(enabled=False)), NullTracer)
    jsonl = build_tracer(ObservabilityConfig(enabled=True, backend="jsonl",
                                             path=str(tmp_path / "t.jsonl")))
    assert isinstance(jsonl, JsonlTracer)
