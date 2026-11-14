"""Pruning unsupported sentences (generation/prune.py, verification.prune_unsupported)."""

from __future__ import annotations

from sec_llm_project.sec_intel.core.config import GenerationConfig, VerificationConfig
from sec_llm_project.sec_intel.core.types import (
    Chunk,
    ClaimVerdict,
    FilingMetadata,
    RetrievalResult,
)
from sec_llm_project.sec_intel.generation.grounded import GroundedAnswerer
from sec_llm_project.sec_intel.generation.prune import prune_unsupported, split_sentences
from sec_llm_project.sec_intel.generation.verifier import CitationVerifier
from sec_llm_project.sec_intel.llm.base import GenerationResult

TEXT = ("The company maintains cyber insurance [1]. The insurance covers all losses [2]. "
        "The excerpts do not discuss valuation.")


def _cv(claim, verdict, meta=False):
    return ClaimVerdict(claim=claim, verdict=verdict, citation=1, meta=meta)


def test_split_keeps_citations_on_their_sentence():
    assert split_sentences(TEXT) == [
        "The company maintains cyber insurance [1].",
        "The insurance covers all losses [2].",
        "The excerpts do not discuss valuation."]


def test_removes_unsupported_sentence_and_keeps_meta():
    claims = [_cv("The company maintains cyber insurance", "SUPPORTED"),
              _cv("The insurance covers all losses", "UNSUPPORTED"),
              _cv("The excerpts do not discuss valuation", "UNSUPPORTED", meta=True)]
    pr = prune_unsupported(TEXT, claims)
    assert pr.text == ("The company maintains cyber insurance [1]. "
                       "The excerpts do not discuss valuation.")
    assert pr.removed_sentences == ["The insurance covers all losses [2]."]
    assert [c.claim for c in pr.kept] == ["The company maintains cyber insurance",
                                          "The excerpts do not discuss valuation"]
    assert pr.collateral == 0


def test_supported_claim_in_a_removed_sentence_is_collateral():
    text = "Revenue grew and margins doubled [1]. Costs fell [1]."
    claims = [_cv("Revenue grew", "SUPPORTED"), _cv("margins doubled", "CONTRADICTED"),
              _cv("Costs fell", "SUPPORTED")]
    pr = prune_unsupported(text, claims)
    assert pr.text == "Costs fell [1]."
    assert pr.collateral == 1


def test_nothing_to_prune_returns_text_unchanged():
    pr = prune_unsupported(TEXT, [_cv("The company maintains cyber insurance", "SUPPORTED")])
    assert not pr.changed and pr.text == TEXT


def _answerer(answer_text, verdicts, prune):
    chunk = Chunk(chunk_id="c0", text="We maintain cyber insurance.",
                  metadata=FilingMetadata(ticker="NOVA"))
    chunk2 = Chunk(chunk_id="c1", text="Other text.", metadata=FilingMetadata(ticker="NOVA"))
    ctx = [RetrievalResult(chunk=chunk, score=1.0, rank=0),
           RetrievalResult(chunk=chunk2, score=0.9, rank=1)]

    class Retriever:
        last_latency_ms = 0.0

        def retrieve(self, question, top_k=None, filters=None):
            return ctx

    class Gen:
        model = "gen"

        def generate(self, messages, **kw):
            return GenerationResult(text=answer_text, model="gen")

    class Judge:
        name = "judge"

        def judge(self, text, context):
            return [ClaimVerdict(claim=c, verdict=v, citation=1,
                                 quote="We maintain cyber insurance" if v == "SUPPORTED" else "")
                    for c, v in verdicts]

    # floor 0.4: one of two claims unsupported still passes the gate
    vcfg = VerificationConfig(prune_unsupported=prune, groundedness_floor=0.4, repair=False)
    return GroundedAnswerer(Retriever(), Gen(), GenerationConfig(min_evidence_score=0.0),
                            verifier=CitationVerifier(Judge(), vcfg))


def test_pipeline_prunes_and_records_what_it_removed():
    text = "The company maintains cyber insurance [1]. The insurance covers all losses [2]."
    verdicts = [("The company maintains cyber insurance", "SUPPORTED"),
                ("The insurance covers all losses", "UNSUPPORTED")]
    ans = _answerer(text, verdicts, prune=True).answer("Cyber insurance?")
    assert not ans.abstained
    assert ans.text == "The company maintains cyber insurance [1]."
    assert [c.chunk_id for c in ans.citations] == ["c0"]  # [2] no longer cited
    assert ans.extra["pruned"]["removed_sentences"] == ["The insurance covers all losses [2]."]
    assert ans.extra["verification"]["total"] == 1
    assert ans.extra["verification"]["groundedness"] == 1.0

    off = _answerer(text, verdicts, prune=False).answer("Cyber insurance?")
    assert off.text == text and "pruned" not in off.extra


def test_split_does_not_break_at_initials_titles_or_company_suffixes():
    text = ("Jonathan P. Graham joined from Danaher Corp. in 2019 [2]. Mr. Brown is CEO [1].\n"
            "Regarding storage:\n*   Capacity rose [3].")
    assert split_sentences(text) == [
        "Jonathan P. Graham joined from Danaher Corp. in 2019 [2].", "Mr. Brown is CEO [1].",
        "Regarding storage:", "*   Capacity rose [3]."]


def test_case_name_abbreviations_and_headings():
    from sec_llm_project.sec_intel.generation.prune import is_lead_in

    text = ("The lawsuit (Waterford Twp. Gen. Emps. Ret. Sys. v. Smith) lacks merit [1].\n"
            "**Risk Profile**\n*   **Technology Integration:**\nThe company may face risk [2].")
    sents = split_sentences(text)
    assert sents[0].endswith("lacks merit [1].") and len(sents) == 4
    assert [is_lead_in(s) for s in sents] == [False, True, True, False]
