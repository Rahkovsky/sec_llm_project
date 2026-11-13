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
    cfg = AppConfig.offline()
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
    cfg = AppConfig.offline()
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


class _ListJudge:
    name = "list"

    def __init__(self, claims):
        self._claims = claims

    def judge(self, answer_text, context):
        return [ClaimVerdict(**c) for c in self._claims]


def test_quote_match_ignores_punctuation_case_and_typography():
    from sec_llm_project.sec_intel.generation.verifier import quote_matches

    src = "The Company\u2019s board — through its Audit Committee — oversees risk, quarterly."
    assert quote_matches("The Company's board - through its Audit Committee - oversees risk", src)
    assert quote_matches("the company's board, through its audit committee, oversees", src)
    # short quotes need an exact match; reordered or absent words never match
    assert not quote_matches("oversees risk quarterly", src)
    assert not quote_matches("The Audit Committee oversees the board through risk", src)


def test_citation_repointed_to_chunk_holding_quote():
    ctx = _ctx(["Unrelated revenue table for the fiscal year.",
                "The Audit Committee reviews cybersecurity risk with the CISO quarterly."])
    judge = _ListJudge([{"claim": "The Audit Committee reviews cyber risk.", "verdict": "SUPPORTED",
                             "citation": 1, "quote": "Audit Committee reviews cybersecurity risk with the CISO"}])
    report = CitationVerifier(judge).verify(Answer(question="q", text="x [1]"), ctx)
    c = report.claims[0]
    assert (c.verdict, c.citation, c.citation_corrected, c.valid_quote) == ("SUPPORTED", 2, True, True)
    assert report.to_dict()["citations_corrected"] == 1


def test_meta_claims_excluded_from_groundedness():
    from sec_llm_project.sec_intel.generation.verifier import is_meta_claim

    assert is_meta_claim("The provided excerpts do not contain information regarding investor sentiment.")
    assert is_meta_claim("The documents do not provide a specific growth or valuation outlook.")
    assert not is_meta_claim("The excerpts state that the CISO reports to the Technology Committee.")
    assert not is_meta_claim("The excerpts do not state that NIST measures cause losses; rather, "
                             "they describe the framework.")
    # Any verb after "do not", with an optional adverb (v3 run phrasings).
    assert is_meta_claim("The excerpts do not cover the impact of executive diversity on growth.")
    assert is_meta_claim("The excerpts do not explicitly state the impact on investor confidence.")
    # Second half of a split "do not contain X, nor do they describe Y" (placebo_0004).
    assert is_meta_claim("nor do they describe any impairment charges related to such an event")
    assert not is_meta_claim("Nor does the company expect material impairment charges.")
    # Describing which filings are in the evidence (placebo_0001).
    assert is_meta_claim("The documents include reports for fiscal years 2021, 2022, and 2024, but "
                         "they do not include a fiscal 2028 10-K.")
    # Positive statements about the filing's content stay scored.
    assert not is_meta_claim("The excerpts do, however, state that new laws may affect pricing.")
    assert not is_meta_claim("The documents also list the names and prior experience of officers.")
    assert not is_meta_claim("The excerpts cover the following risks: supply chain and cyber.")
    ctx = _ctx(["The Audit Committee reviews cybersecurity risk with the CISO quarterly."])
    judge = _ListJudge([
        {"claim": "The Audit Committee reviews cyber risk.", "verdict": "SUPPORTED", "citation": 1,
             "quote": "The Audit Committee reviews cybersecurity risk"},
        {"claim": "The provided excerpts do not contain information regarding investor sentiment.",
             "verdict": "UNSUPPORTED"}])
    report = CitationVerifier(judge).verify(Answer(question="q", text="x [1]"), ctx)
    assert (report.total, report.groundedness, report.to_dict()["meta"]) == (1, 1.0, 1)


def test_grouped_citation_markers_parsed():
    from sec_llm_project.sec_intel.generation.verifier import cited_numbers

    assert cited_numbers("A [1, 2]. B [3]. C [2;4]. D [5-6].") == [1, 2, 3, 4, 5, 6]


def test_insufficient_flag_only_abstains_when_bare():
    from sec_llm_project.sec_intel.generation.grounded import strip_insufficient

    assert strip_insufficient("INSUFFICIENT_EVIDENCE") == (None, False)
    assert strip_insufficient("The excerpts do not cover this.\n\nINSUFFICIENT_EVIDENCE") == (None, False)
    text, partial = strip_insufficient(
        "The Board oversees enterprise risk through the Risk and Safety Committee [1, 2].\n"
        "INSUFFICIENT_EVIDENCE for the effect on investor confidence.")
    assert partial and "INSUFFICIENT_EVIDENCE" not in text and "[1, 2]" in text
    assert strip_insufficient("Revenue was $5 billion [1].") == ("Revenue was $5 billion [1].", False)


def test_qualifier_candidates_and_check_parsing():
    from sec_llm_project.sec_intel.generation.qualifiers import is_candidate, parse_check

    quote = "The emergence of LLMs may further intensify these pressures."
    assert is_candidate("Competition is intensified by LLMs.", quote) == ["may"]
    assert is_candidate("LLMs may intensify competition.", quote) == []
    assert is_candidate("Revenue rose 5%.", "Revenue rose 5%.") == []
    assert parse_check("1: DROPPED\n2: OK\n**3: dropped**\n- 4. OK", 4) == [True, False, True, False]
    assert parse_check("garbage", 2) == [None, None]


def test_qualifier_repair_restores_hedge_and_records_it():
    from sec_llm_project.sec_intel.core.config import GenerationConfig
    from sec_llm_project.sec_intel.generation.grounded import GroundedAnswerer
    from sec_llm_project.sec_intel.llm.base import GenerationResult

    quote = "competitive pressures could harm our business and financial performance"
    ctx = _ctx([f"We operate in an intensely competitive industry and {quote}."])

    class Retriever:
        last_latency_ms = 0.0

        def retrieve(self, question, top_k=None, filters=None):
            return ctx

    class Gen:  # first answer hardens "could harm"; the repair restores it
        model = "gen"

        def __init__(self):
            self.calls = []

        def generate(self, messages, **kw):
            self.calls.append(messages)
            text = ("Competitive pressures can harm the business [1]." if len(self.calls) == 1
                    else "Competitive pressures could harm the business [1].")
            return GenerationResult(text=text, model="gen")

    class Judge:  # every answer sentence is supported by the quote
        name = "judge"

        def judge(self, answer_text, context):
            claim = answer_text.replace(" [1]", "")
            return [ClaimVerdict(claim=claim, verdict="SUPPORTED", citation=1, quote=quote)]

    class Checker:  # DROPPED when the claim lacks the filing's "could"
        def check(self, items):
            return ["could" not in it.claim for it in items]

    cfg = VerificationConfig(qualifier_check=True, qualifier_repair=True)
    verifier = CitationVerifier(Judge(), cfg, qualifier_checker=Checker())
    gen = Gen()
    ans = GroundedAnswerer(Retriever(), gen, GenerationConfig(min_evidence_score=0.0),
                           verifier=verifier).answer("How does competition affect HP?")
    assert not ans.abstained
    assert "could harm" in ans.text
    q = ans.extra["qualifiers"]
    assert q["repaired"] and q["dropped"] == [] and len(q["dropped_before"]) == 1
    # The repair prompt quotes the claim and the filing text.
    assert "can harm" in gen.calls[1][-1].content and quote in gen.calls[1][-1].content


def test_qualifier_rule_is_opt_in():
    from sec_llm_project.sec_intel.generation import prompts

    assert prompts.system_prompt() == prompts.SYSTEM_PROMPT
    assert "qualifiers" in prompts.system_prompt(True)
    assert "one sentence per point" in prompts.system_prompt(True, True)
    assert "one sentence per point" not in prompts.system_prompt(True)
