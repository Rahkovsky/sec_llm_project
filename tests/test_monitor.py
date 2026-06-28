from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.evaluation.dataset import index_sample_corpus
from sec_llm_project.sec_intel.monitor.signals import detect_signals
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline


def _pipeline():
    cfg = AppConfig()
    cfg.audit.enabled = False
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    return p


def test_monitor_detects_new_risks_and_signals():
    p = _pipeline()
    report = p.monitor("NOVA", "2023", "2024").to_dict()
    s = report["summary"]
    assert s["new_risk_factors"] >= 2          # AI regulation + interest-rate risks
    assert s["new_litigation_disclosures"] >= 1
    assert s["risk_signals"] >= 1
    # the material weakness must surface as a grounded signal
    assert "material_weakness" in report["signals"]
    mw = report["signals"]["material_weakness"][0]
    assert mw["citation"]["chunk_id"]
    assert mw["citation"]["item_number"] == "9A"


def test_signal_detection_is_grounded():
    p = _pipeline()
    chunks = [c for c in p.index.all_chunks() if c.metadata.ticker == "NOVA"]
    signals = detect_signals(chunks)
    assert signals
    for sig in signals:
        assert sig.citation.quote          # every signal carries a quote
        assert sig.citation.chunk_id
