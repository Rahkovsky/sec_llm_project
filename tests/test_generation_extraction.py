from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.evaluation.dataset import index_sample_corpus
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline


def _pipeline():
    cfg = AppConfig()
    cfg.audit.enabled = False
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    return p


def test_grounded_answer_has_citations():
    p = _pipeline()
    ans = p.ask("What supply chain risks does Nova Devices face?", filters={"ticker": "NOVA"})
    assert not ans.abstained
    assert ans.citations
    assert ans.confidence > 0
    # at least one citation comes from the Risk Factors section
    assert any(c.item_number == "1A" for c in ans.citations)


def test_offtopic_question_abstains():
    p = _pipeline()
    ans = p.ask("What is the airspeed velocity of an unladen swallow?",
                filters={"ticker": "NOVA"})
    assert ans.abstained
    assert ans.confidence == 0.0


def test_extraction_returns_valid_schema():
    p = _pipeline()
    res = p.extract("risk_factors", ticker="NOVA")
    assert res.valid
    assert "risk_factors" in res.data
    assert isinstance(res.data["risk_factors"], list)
    assert res.citations
