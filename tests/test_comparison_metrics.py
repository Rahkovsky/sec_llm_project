from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.evaluation import metrics
from sec_llm_project.sec_intel.evaluation.dataset import CURATED_CASES, index_sample_corpus
from sec_llm_project.sec_intel.evaluation.runner import EvaluationRunner
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline


def test_recall_is_set_based():
    # duplicates must not push recall above 1.0
    assert metrics.recall_at_k(["1A", "1A", "1A"], ["1A"], 8) == 1.0
    assert metrics.recall_at_k(["7", "3"], ["1A"], 8) == 0.0


def test_mrr_and_map():
    assert metrics.reciprocal_rank(["x", "y", "rel"], ["rel"]) == 1 / 3
    assert metrics.mrr([["rel"], ["x", "rel"]], [["rel"], ["rel"]]) == (1.0 + 0.5) / 2
    ap = metrics.average_precision_at_k(["rel", "x", "rel2"], ["rel", "rel2"], 8)
    assert 0 < ap <= 1


def test_filing_comparison_detects_changes():
    cfg = AppConfig()
    cfg.audit.enabled = False
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    report = p.compare_years("NOVA", "2023", "2024", items=["1A", "3", "9A"]).to_dict()
    assert report["summary"]["added"] >= 1
    assert report["summary"]["removed"] >= 1
    # The new material weakness should appear among Item 9A additions.
    item9a = next(s for s in report["sections"] if s["item_number"] == "9A")
    assert any("material weakness" in a.lower() for a in item9a["added"])


def test_evaluation_runner_produces_metrics():
    cfg = AppConfig()
    cfg.audit.enabled = False
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    report = EvaluationRunner(p).run(CURATED_CASES, k=8)
    assert 0.0 <= report.metrics["recall@k"] <= 1.0
    assert 0.0 <= report.metrics["mrr"] <= 1.0
    assert report.metrics["abstention_recall"] == 1.0  # unanswerable case abstains
