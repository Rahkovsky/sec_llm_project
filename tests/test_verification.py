"""Citation verification: lexical judge, the verbatim-quote invariant, the
offline/real judge selection, and end-to-end attachment via the pipeline.
"""

from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig, VerificationConfig
from sec_llm_project.sec_intel.core.types import (
    Answer,
    Chunk,
    ClaimVerdict,
    FilingMetadata,
    RetrievalResult,
)
from sec_llm_project.sec_intel.evaluation.dataset import index_sample_corpus
from sec_llm_project.sec_intel.generation.verifier import (
    CitationVerifier,
    LexicalJudge,
    build_verifier,
)
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline


def _ctx(texts):
    out = []
    for i, t in enumerate(texts):
        chunk = Chunk(chunk_id=f"c{i}", text=t,
                      metadata=FilingMetadata(ticker="NOVA"),
                      section_title="Risk Factors", item_number="1A")
        out.append(RetrievalResult(chunk=chunk, score=1.0, rank=i))
    return out


def test_lexical_judge_supports_grounded_claim():
    ctx = _ctx(["Nova Devices faces supply chain concentration risk in a single region."])
    answer = Answer(question="q", text="Nova Devices faces supply chain concentration risk. [1]")
    report = CitationVerifier(LexicalJudge()).verify(answer, ctx)
    assert report.total >= 1
    assert report.supported >= 1
    assert report.groundedness > 0.5
    # The verbatim quote really comes from the cited chunk.
    assert all(c.valid_quote for c in report.claims if c.citation is not None)


def test_unsupported_claim_is_flagged():
    ctx = _ctx(["Nova Devices designs and sells precision sensors to customers."])
    answer = Answer(question="q",
                    text="Nova Devices disclosed a going concern doubt and a bankruptcy petition. [1]")
    report = CitationVerifier(LexicalJudge()).verify(answer, ctx)
    assert any(c.verdict == "UNSUPPORTED" for c in report.claims)
    assert report.groundedness < 0.6


def test_substring_check_downgrades_fabricated_quote():
    """A judge that asserts SUPPORTED with a quote not in the source is overruled."""

    class FakeJudge:
        name = "fake"

        def judge(self, answer_text, context):
            return [ClaimVerdict(claim="x", verdict="SUPPORTED", citation=1,
                                 quote="this exact text does not appear in the evidence")]

    ctx = _ctx(["Completely unrelated content about quarterly revenue."])
    report = CitationVerifier(FakeJudge()).verify(Answer(question="q", text="x [1]"), ctx)
    assert report.claims[0].verdict == "UNSUPPORTED"
    assert report.claims[0].valid_quote is False


def test_extract_and_match_figures():
    from sec_llm_project.sec_intel.generation.numeric import (
        extract_figures,
        figures_supported,
    )

    # Scale folding: "$1,577 million" -> 1.577e9; percent stays its own class.
    figs = extract_figures("Capex was $1,577 million and margin was 14%.")
    assert any(abs(f.value - 1.577e9) < 1 and not f.is_percent for f in figs)
    assert any(abs(f.value - 14.0) < 1e-9 and f.is_percent for f in figs)

    # Implicit table scaling: claim in billions vs evidence bare millions.
    assert figures_supported("Debt was $2.5 billion", "Total debt 2,500 (in millions)")
    # Wrong figure is not supported.
    assert not figures_supported("Debt was $3.1 billion", "Total debt was $2.5 billion")
    # Percent must match a percent, not a bare number.
    assert figures_supported("Margin was 14%", "gross margin of 14.0% this year")
    assert not figures_supported("Margin was 14%", "we shipped 14 units")
    # No figures in the claim -> nothing to refute.
    assert figures_supported("Revenue grew during the year", "anything at all")


def test_numeric_invariant_downgrades_wrong_figure():
    """A SUPPORTED claim whose figure is absent from the cited chunk is overruled,
    even when its quote is a valid substring (numeric hallucination guard)."""

    class FigureJudge:
        name = "fig"

        def __init__(self, claim):
            self._claim = claim

        def judge(self, answer_text, context):
            return [ClaimVerdict(claim=self._claim, verdict="SUPPORTED", citation=1,
                                 quote="cash from operations was 6,439")]

    ctx = _ctx(["Net cash from operations was 6,439 for the year."])

    # Correct figure -> stays supported.
    ok = CitationVerifier(FigureJudge("Operating cash flow was 6,439")).verify(
        Answer(question="q", text="x [1]"), ctx)
    assert ok.claims[0].verdict == "SUPPORTED"
    assert ok.claims[0].numeric_ok is True

    # Fabricated figure (quote still valid) -> downgraded.
    bad = CitationVerifier(FigureJudge("Operating cash flow was 9,999")).verify(
        Answer(question="q", text="x [1]"), ctx)
    assert bad.claims[0].verdict == "UNSUPPORTED"
    assert bad.claims[0].numeric_ok is False


def test_build_verifier_selects_lexical_offline_and_none_when_disabled():
    verifier = build_verifier(VerificationConfig(enabled=True))  # judge defaults to mock
    assert verifier is not None
    assert isinstance(verifier.judge, LexicalJudge)
    assert build_verifier(VerificationConfig(enabled=False)) is None


def test_build_verifier_rejects_self_grading_judge():
    """An LLM judge that is the same model as the generator is not independent."""
    import pytest

    from sec_llm_project.sec_intel.core.config import LLMConfig

    gen = LLMConfig(provider="ollama", model="gemma3:12b")
    vcfg = VerificationConfig(
        enabled=True, judge=LLMConfig(provider="ollama", model="gemma3:12b")
    )
    with pytest.raises(ValueError, match="independence"):
        build_verifier(vcfg, generator=gen)

    # A different local model as judge is allowed (no network call until used).
    vcfg_ok = VerificationConfig(
        enabled=True, judge=LLMConfig(provider="ollama", model="mistral:7b")
    )
    assert build_verifier(vcfg_ok, generator=gen) is not None


def test_pipeline_attaches_verification_report():
    cfg = AppConfig()
    cfg.audit.enabled = False
    cfg.verification.enabled = True
    cfg.verification.groundedness_floor = 0.0  # attach the report without gating
    pipeline = SECIntelPipeline(cfg)
    index_sample_corpus(pipeline)

    answer = pipeline.ask("What supply chain risks does Nova Devices face?",
                          filters={"ticker": "NOVA"})
    assert "verification" in answer.extra
    report = answer.extra["verification"]
    assert report["total"] >= 1
    assert 0.0 <= report["groundedness"] <= 1.0
    assert report["judge_model"] == "lexical-judge"


def test_pipeline_gates_when_groundedness_floor_is_high():
    cfg = AppConfig()
    cfg.audit.enabled = False
    cfg.verification.enabled = True
    cfg.verification.groundedness_floor = 1.01  # impossible to satisfy -> must abstain
    cfg.verification.repair = False
    pipeline = SECIntelPipeline(cfg)
    index_sample_corpus(pipeline)

    answer = pipeline.ask("What supply chain risks does Nova Devices face?",
                          filters={"ticker": "NOVA"})
    assert answer.abstained
    assert "verification" in answer.extra
