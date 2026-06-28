from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.evaluation.dataset import index_sample_corpus
from sec_llm_project.sec_intel.evaluation.enforcement import (
    ENFORCEMENT_CASES,
    run_enforcement_benchmark,
)
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline


def test_enforcement_benchmark_runs_offline():
    cfg = AppConfig()
    cfg.audit.enabled = False
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    report = run_enforcement_benchmark(p, ENFORCEMENT_CASES)
    assert report.n_cases == len(ENFORCEMENT_CASES)
    assert 0.0 <= report.metrics["signal_recall"] <= 1.0
    assert 0.0 <= report.metrics["evidence_retrieval_recall"] <= 1.0
    # the material-weakness case should detect its signal in the corpus
    mw = next(c for c in report.per_case if c["id"] == "rev-rec-material-weakness")
    assert "material_weakness" in mw["found_signals"]
    # every case references a public SEC enforcement resource
    assert all(c["source_url"].startswith("https://www.sec.gov") for c in report.per_case)
