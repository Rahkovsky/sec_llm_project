from __future__ import annotations

import pytest

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.evaluation.dataset import index_sample_corpus
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline
from sec_llm_project.sec_intel.provenance import ProvenanceDAG, build_dag


def _answer():
    cfg = AppConfig.offline()
    cfg.verification.enabled = True  # LexicalJudge (deterministic, offline)
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    return p.ask("What supply chain risks does Nova Devices face?",
                 filters={"ticker": "NOVA"})


def test_dag_from_pipeline_answer():
    dag = build_dag(_answer().to_dict())
    types = {n.type for n in dag.nodes}
    assert "question" in types and "evidence_span" in types
    assert types & {"final_conclusion", "rejected_conclusion"}
    # every claim connects to the conclusion; supported claims have evidence
    claims = [n for n in dag.nodes if n.type == "atomic_claim"]
    assert claims
    for c in claims:
        assert any(e.source == c.id and e.type == "part_of_answer" for e in dag.edges)
    rows = dag.mistake_table()
    assert len(rows) == len(claims)
    s = dag.summary()
    assert s["claims"] == len(claims) and s["weighted_claim_error_rate"] is not None
    assert "digraph" in dag.to_dot()


def test_abstained_answer_builds_rejected_conclusion_without_report():
    a = {"question": "q", "text": "no evidence", "abstained": True,
         "abstain_reason": "weak evidence", "citations": [], "extra": {}}
    dag = build_dag(a)
    assert dag.node("a").type == "rejected_conclusion"
    assert dag.summary()["rejected_conclusion"] == "weak evidence"


def test_from_dict_roundtrip_and_html():
    dag = build_dag(_answer().to_dict())
    back = ProvenanceDAG.from_dict(dag.to_dict())
    assert len(back.nodes) == len(dag.nodes) and len(back.edges) == len(dag.edges)
    html = dag.to_html("t")
    assert "vis-network" in html and '"id": "a"' in html
    assert "supply chain" in html.lower()  # question surfaces in the header


def test_dag_requires_verification():
    cfg = AppConfig.offline()  # verification off -> no report -> refuse
    p = SECIntelPipeline(cfg)
    index_sample_corpus(p)
    a = p.ask("What supply chain risks does Nova Devices face?", filters={"ticker": "NOVA"})
    with pytest.raises(ValueError, match="no verification"):
        build_dag(a.to_dict())
