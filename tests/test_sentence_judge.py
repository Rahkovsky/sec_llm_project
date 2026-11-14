"""Compact judge formats (verification.judge_format: sentences / verdicts)."""

from __future__ import annotations

import json

from sec_llm_project.sec_intel.core.config import VerificationConfig
from sec_llm_project.sec_intel.core.types import Answer, Chunk, FilingMetadata, RetrievalResult
from sec_llm_project.sec_intel.generation.verifier import (
    CitationVerifier,
    SentenceLLMJudge,
    parse_sentence_verdicts,
)
from sec_llm_project.sec_intel.llm.base import GenerationResult

SENTS = ["Revenue was $5.2 billion [1, 2].", "The company may face cyber risk [2].",
         "The excerpts do not discuss valuation."]


def _ctx():
    texts = ["Total revenue was $5.2 billion in 2024.",
             "Revenue grew 4%. We may face risks from cyber attacks on our systems."]
    return [RetrievalResult(chunk=Chunk(chunk_id=f"c{i}", text=t,
                                        metadata=FilingMetadata(ticker="NOVA")),
                            score=1.0, rank=i) for i, t in enumerate(texts)]


def test_parse_one_verdict_per_sentence_and_strip_markers():
    raw = json.dumps({"sentences": [
        {"s": 1, "verdict": "SUPPORTED", "citation": 1, "quote": "Total revenue was $5.2 billion"},
        {"s": 2, "verdict": "bogus", "citation": 9},
    ]})
    out = parse_sentence_verdicts(raw, SENTS, n_evidence=2)
    assert [c.claim for c in out] == ["Revenue was $5.2 billion.",
                                      "The company may face cyber risk.",
                                      "The excerpts do not discuss valuation."]
    assert out[0].sources == [1, 2] and out[0].citation == 1
    assert out[1].verdict == "UNSUPPORTED" and out[1].citation is None  # bad verdict/citation
    assert out[2].verdict == "UNSUPPORTED"  # skipped by the judge


def test_malformed_judge_output_marks_every_sentence_unsupported():
    out = parse_sentence_verdicts("not json", SENTS, n_evidence=2)
    assert [c.verdict for c in out] == ["UNSUPPORTED"] * 3


class _LLM:
    model = "judge"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, messages, **kw):
        return GenerationResult(text=json.dumps(self.payload), model="judge")


def test_verdicts_format_takes_quote_from_cited_passage():
    llm = _LLM({"sentences": [{"s": 1, "verdict": "SUPPORTED", "citation": 1},
                              {"s": 2, "verdict": "SUPPORTED", "citation": 2},
                              {"s": 3, "verdict": "SUPPORTED", "citation": None}]})
    verifier = CitationVerifier(SentenceLLMJudge(llm, quotes=False), VerificationConfig())
    report = verifier.verify(Answer(question="q", text=" ".join(SENTS)), _ctx())
    c1, c2, c3 = report.claims
    assert c1.verdict == "SUPPORTED" and c1.valid_quote
    assert c2.quote == "We may face risks from cyber attacks on our systems."
    assert c3.meta  # meta sentence: not scored
    assert report.total == 2 and report.groundedness == 1.0


def test_figure_check_uses_every_passage_the_sentence_cites():
    # "$5.2 billion" is only in passage 1; the judge cites 2, the sentence cites [1, 2]
    llm = _LLM({"sentences": [{"s": 1, "verdict": "SUPPORTED", "citation": 2,
                               "quote": "Revenue grew 4%."}]})
    verifier = CitationVerifier(SentenceLLMJudge(llm), VerificationConfig())
    report = verifier.verify(Answer(question="q", text=SENTS[0]), _ctx())
    assert report.claims[0].numeric_ok and report.claims[0].verdict == "SUPPORTED"


def test_lead_in_line_is_not_scored():
    text = "Regarding risk:\n*   The company may face cyber risk [2]."
    llm = _LLM({"sentences": [{"s": 1, "verdict": "UNSUPPORTED", "citation": None},
                              {"s": 2, "verdict": "SUPPORTED", "citation": 2}]})
    verifier = CitationVerifier(SentenceLLMJudge(llm, quotes=False), VerificationConfig())
    report = verifier.verify(Answer(question="q", text=text), _ctx())
    assert report.claims[0].lead_in and report.claims[0].meta
    assert report.total == 1 and report.groundedness == 1.0
