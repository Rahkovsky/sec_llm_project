"""Faithfulness metrics in the evaluation runner + report-only snapshot diffing."""

from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.evaluation.dataset import CURATED_CASES, index_sample_corpus
from sec_llm_project.sec_intel.evaluation.runner import EvaluationRunner
from sec_llm_project.sec_intel.evaluation.snapshot import (
    diff_metrics,
    load_baseline,
    render_diff,
    save_baseline,
)
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline


def _pipeline():
    cfg = AppConfig()
    cfg.audit.enabled = False
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    return p


def test_runner_reports_faithfulness_metrics():
    report = EvaluationRunner(_pipeline()).run(CURATED_CASES, k=8, generate=True)
    m = report.metrics
    # Production verification is off here, so the runner falls back to its lexical
    # judge. The metric NAMES must reflect that: groundedness is reported as
    # `lexical_groundedness` (keyword overlap, not entailment) and contradiction_rate
    # is omitted entirely (the lexical judge cannot detect contradictions).
    assert report.judge_model == "lexical-judge"
    assert "lexical_groundedness" in m and "mean_groundedness" not in m
    assert "contradiction_rate" not in m
    for key in ("lexical_groundedness", "fully_grounded_rate"):
        assert 0.0 <= m[key] <= 1.0
    # Retrieval metrics still present and unchanged in shape.
    assert "recall@k" in m and "citation_correctness" in m


def test_no_faithfulness_without_generation():
    report = EvaluationRunner(_pipeline()).run(CURATED_CASES, k=8, generate=False)
    assert "mean_groundedness" not in report.metrics
    assert "lexical_groundedness" not in report.metrics
    assert "recall@k" in report.metrics


def test_snapshot_roundtrip_and_no_regression(tmp_path):
    report = {"k": 8, "n_cases": 7,
              "metrics": {"mean_groundedness": 0.9, "mean_latency_ms": 12.3}}
    path = tmp_path / "baseline.json"
    save_baseline(report, path)

    loaded = load_baseline(path)
    assert loaded is not None
    # Latency is environment-dependent and must be excluded from the baseline.
    assert "mean_latency_ms" not in loaded["metrics"]
    assert loaded["metrics"]["mean_groundedness"] == 0.9

    diff = diff_metrics({"metrics": {"mean_groundedness": 0.905}}, loaded)
    assert diff["regressions"] == []


def test_snapshot_detects_regression():
    base = {"metrics": {"mean_groundedness": 0.9, "hallucination_rate": 0.0}}
    cur = {"metrics": {"mean_groundedness": 0.70, "hallucination_rate": 0.25}}
    diff = diff_metrics(cur, base)
    # A drop in a higher-is-better metric and a rise in a lower-is-better one.
    assert "mean_groundedness" in diff["regressions"]
    assert "hallucination_rate" in diff["regressions"]
    assert "REGRESSION" in render_diff(diff)


def test_missing_baseline_returns_none(tmp_path):
    assert load_baseline(tmp_path / "does_not_exist.json") is None
