"""Citation verification via an independent LLM-as-judge (plan: responsible AI).

Every factual claim in a grounded answer is checked against the cited evidence by
a judge configured *separately* from the generator (cross-vendor by default), so
the generator never grades its own work. Two safeguards back the judge:

* a deterministic substring check — a "supporting" quote must actually occur in
  the cited chunk, which catches a judge that fabricates evidence; and
* an offline lexical-overlap judge, so the whole verification path runs in
  tests/CI without any API keys (mirroring the hashing-embeddings fallback).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Protocol, runtime_checkable

from ..core.config import LLMConfig, VerificationConfig
from ..core.types import (
    CONTRADICTED,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    Answer,
    ClaimVerdict,
    RetrievalResult,
    VerificationReport,
)
from ..llm.base import LLMProvider, Message
from . import prompts
from .numeric import figures_supported

# Citation markers: [3], and the grouped forms the generator also writes, [1, 2],
# [1,2], [2; 4], [1-3]. (A [n]-only parser missed every grouped marker: 24 of 266 v2
# answers cited only in grouped form, so their citations fell back to excerpts 1-2.)
_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;\u2013-]\s*\d+)*)\]")


def cited_numbers(text: str) -> list[int]:
    """Excerpt numbers cited in `text`, in order of first appearance."""
    out: list[int] = []
    for m in _CITE_RE.finditer(text):
        for part in re.split(r"\s*[,;]\s*", m.group(1)):
            ends = [int(x) for x in re.split(r"\s*[\u2013-]\s*", part)]
            lo, hi = ends[0], ends[-1]
            for n in range(lo, hi + 1) if 0 < hi - lo < 20 else ends:
                if n not in out:
                    out.append(n)
    return out


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "are", "was", "were", "be", "by", "as", "at", "that", "this", "from", "it",
    "its", "we", "our", "their", "have", "has", "had", "which", "these", "those",
    "what", "how", "did", "does", "do", "than", "into", "about", "also",
}
_VERDICTS = {SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED}


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


_TYPO = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                       "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " "})
_QWORD_RE = re.compile(r"[a-z0-9]+")
MIN_FUZZY_QUOTE_WORDS = 5


def _qwords(text: str) -> str:
    """Word stream for quote matching: case, punctuation, quote style, dashes ignored."""
    return " " + " ".join(_QWORD_RE.findall(unicodedata.normalize("NFKC", text)
                                            .translate(_TYPO).lower())) + " "


def quote_matches(quote: str, source: str) -> bool:
    """A judge's quote occurs in the source. Exact (whitespace-normalized) always
    counts; otherwise a quote of >= 5 words counts when its word sequence occurs
    contiguously in the source, ignoring punctuation, case, curly quotes and dashes.
    In the v2 run 57 of 126 'unsupported' claims carried a correct quote that the
    whitespace-only check rejected for a comma, a period or a curly quote."""
    if not quote.strip():
        return False
    if _normalize_ws(quote) in _normalize_ws(source):
        return True
    q = _qwords(quote)
    return len(q.split()) >= MIN_FUZZY_QUOTE_WORDS and q in _qwords(source)


# "The provided excerpts do not contain information regarding ..." is a statement
# about the evidence, not a claim about the company: it can be neither supported
# nor refuted by a quote, so it is excluded from groundedness (11 of 126 v2
# unsupported claims).
_META_SUBJECT = (r"^(?:however,?\s*)?(?:the\s+)?(?:provided\s+|given\s+|available\s+)?"
                 r"(?:excerpts?|documents?|context|evidence|filings?|text|sources?|"
                 r"information provided)(?:\s+provided)?\s+")
# "The excerpts do not <verb> ..." with any verb ("cover", "explicitly state", ...):
# a statement about what the evidence lacks, not about the company. The v3 prompt
# asks for one sentence on what is not covered, and gemma4 phrases it as "The
# excerpts do not cover ..." (95 of 261 unflagged claims in the v3 run, Nov 12 2026).
_META_RE = re.compile(_META_SUBJECT + r"(?:do|does|did)\s+not\s+(?:\w+ly\s+)?\w+",
                      re.IGNORECASE)
# "The documents include reports for fiscal years 2021, 2022 and 2024, but ..."
# describes which filings are in the evidence, not what they say (placebo_0001).
_META_INVENTORY_RE = re.compile(
    _META_SUBJECT + r"(?:include|includes|cover|covers|contain|contains|consist of|span|are)\s+"
    r"(?:the\s+|only\s+)?(?:annual\s+|quarterly\s+)?(?:reports?|filings?|10-[KQ]s?|"
    r"documents?)\s+(?:for|from|covering)\s+(?:the\s+)?(?:fiscal\s+)?(?:years?|periods?|"
    r"FY|(?:19|20)\d\d)", re.IGNORECASE)
# The judge splits "The excerpts do not contain X, nor do they describe Y" into
# two claims; the second ("nor do they describe Y") is meta too (placebo_0004, v3_gold).
_META_NOR_RE = re.compile(
    r"^(?:and\s+|but\s+)?nor\s+(?:do|does|did)\s+(?:they|it|the\s+(?:provided\s+)?"
    r"(?:excerpts?|documents?|filings?|context|evidence))\s+(?:\w+ly\s+)?\w+", re.IGNORECASE)


def is_meta_claim(claim: str) -> bool:
    c = claim.strip().strip("*-• ")
    if _META_INVENTORY_RE.match(c) or _META_NOR_RE.match(c):
        return True
    return bool(_META_RE.match(c)) and not re.search(r"(?i)\brather\b|;\s*(?:they|it)\b", c)


@runtime_checkable
class Judge(Protocol):
    """Decompose an answer into claims and judge each against the evidence."""

    name: str

    def judge(self, answer_text: str, context: list[RetrievalResult]) -> list[ClaimVerdict]:
        ...


class LexicalJudge:
    """Deterministic, offline judge using keyword overlap (no network/keys).

    Cannot detect CONTRADICTED (that needs a real model); it emits SUPPORTED /
    PARTIAL / UNSUPPORTED so the pipeline is fully exercisable in CI.
    """

    name = "lexical-judge"

    def __init__(self, support: float = 0.6, partial: float = 0.3) -> None:
        self.support = support
        self.partial = partial

    def judge(self, answer_text: str, context: list[RetrievalResult]) -> list[ClaimVerdict]:
        evidence = [(i + 1, _keywords(r.chunk.text), r.chunk.text) for i, r in enumerate(context)]
        claims: list[ClaimVerdict] = []
        for sent in _sentences(_CITE_RE.sub("", answer_text)):
            kw = _keywords(sent)
            if not kw:
                continue
            best_n, best_overlap, best_text = None, 0.0, ""
            for num, ekw, etext in evidence:
                overlap = len(kw & ekw) / len(kw)
                if overlap > best_overlap:
                    best_n, best_overlap, best_text = num, overlap, etext
            if best_overlap >= self.support:
                verdict = SUPPORTED
            elif best_overlap >= self.partial:
                verdict = PARTIAL
            else:
                verdict = UNSUPPORTED
            quote = self._best_quote(kw, best_text) if verdict != UNSUPPORTED else ""
            claims.append(ClaimVerdict(
                claim=sent, verdict=verdict,
                citation=best_n if verdict != UNSUPPORTED else None, quote=quote,
            ))
        return claims

    @staticmethod
    def _best_quote(claim_kw: set[str], evidence_text: str) -> str:
        """The single evidence sentence with the most overlap (verbatim substring)."""
        best, best_score = "", 0.0
        for sent in _sentences(evidence_text):
            skw = _keywords(sent)
            if not skw:
                continue
            score = len(claim_kw & skw) / len(skw)
            if score > best_score:
                best, best_score = sent, score
        return best


class LLMJudge:
    """LLM-backed judge for real (cross-vendor) verification."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.name = str(getattr(llm, "model", None) or getattr(llm, "name", "llm-judge"))

    def judge(self, answer_text: str, context: list[RetrievalResult]) -> list[ClaimVerdict]:
        messages = [
            Message("system", prompts.JUDGE_SYSTEM_PROMPT),
            Message("user", prompts.build_judge_prompt(answer_text, context)),
        ]
        result = self.llm.generate(messages, temperature=0.0, json_schema=prompts.JUDGE_SCHEMA)
        return _parse_claims(result.text, n_evidence=len(context))


def _parse_claims(text: str, *, n_evidence: int) -> list[ClaimVerdict]:
    """Parse a judge's JSON response defensively; malformed output -> no claims."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    out: list[ClaimVerdict] = []
    for raw in data.get("claims", []):
        if not isinstance(raw, dict) or not str(raw.get("claim", "")).strip():
            continue  # an entry with no claim text is judge noise, not a claim
        verdict = str(raw.get("verdict", "")).upper()
        if verdict not in _VERDICTS:
            verdict = UNSUPPORTED
        citation = raw.get("citation")
        if not (isinstance(citation, int) and 1 <= citation <= n_evidence):
            citation = None
        out.append(ClaimVerdict(
            claim=str(raw.get("claim", "")).strip(),
            verdict=verdict,
            citation=citation,
            quote=str(raw.get("quote", "")).strip(),
        ))
    return out


class CitationVerifier:
    """Run a judge over an answer, then enforce the verbatim-quote invariant."""

    def __init__(self, judge: Judge, config: VerificationConfig | None = None,
                 qualifier_checker: Any = None) -> None:
        self.judge = judge
        self.config = config or VerificationConfig()
        # generation.qualifiers.QualifierChecker (or None): judge-model check of
        # supported claims that drop the filing's hedges.
        self.qualifier_checker = qualifier_checker

    def verify(self, answer: Answer, context: list[RetrievalResult]) -> VerificationReport:
        claims = self.judge.judge(answer.text, context)
        for cv in claims:
            cv.meta = is_meta_claim(cv.claim)
            self._repoint_citation(cv, context)
            cv.valid_quote = self._quote_in_source(cv, context)
            cv.numeric_ok = self._numeric_in_source(cv, context)
            # A claim asserted as supported is downgraded if either invariant fails:
            # the supporting quote isn't actually in the cited source, or the claim
            # asserts a figure that does not appear in the cited source (a numeric
            # hallucination that surface-overlap/entailment judges tend to miss).
            # Support must name its source: SUPPORTED/PARTIAL requires a citation
            # whose quote/figures actually check out, else it is unverifiable.
            if cv.verdict in (SUPPORTED, PARTIAL) and (
                cv.citation is None or not cv.valid_quote or not cv.numeric_ok
            ):
                cv.verdict = UNSUPPORTED
            # A contradiction must point at the refuting evidence; "contradicted
            # by nothing" is judge shorthand for absent-from-evidence.
            if cv.verdict == CONTRADICTED and cv.citation is None:
                cv.verdict = UNSUPPORTED
        return VerificationReport(claims=claims, judge_model=self.judge.name)

    @staticmethod
    def _repoint_citation(cv: ClaimVerdict, context: list[RetrievalResult]) -> None:
        """If the quote is not in the cited chunk (or nothing is cited) but occurs in
        another retrieved chunk, point the citation there and flag it. The judge
        numbers excerpts by hand and is off by one now and then (9 v2 claims); the
        quote, not the number, is the evidence."""
        if not cv.quote or cv.verdict not in (SUPPORTED, PARTIAL):
            return
        idx = (cv.citation or 0) - 1
        if 0 <= idx < len(context) and quote_matches(cv.quote, context[idx].chunk.text):
            return
        for j, r in enumerate(context):
            if j != idx and quote_matches(cv.quote, r.chunk.text):
                cv.citation, cv.citation_corrected = j + 1, True
                return

    @staticmethod
    def _quote_in_source(cv: ClaimVerdict, context: list[RetrievalResult]) -> bool:
        if cv.citation is None or not cv.quote:
            return False
        idx = cv.citation - 1
        if not (0 <= idx < len(context)):
            return False
        return quote_matches(cv.quote, context[idx].chunk.text)

    @staticmethod
    def _numeric_in_source(cv: ClaimVerdict, context: list[RetrievalResult]) -> bool:
        """True if every figure in the claim appears in the cited chunk (or the
        claim has no figures / no usable citation — nothing to refute)."""
        if cv.citation is None:
            return True
        idx = cv.citation - 1
        if not (0 <= idx < len(context)):
            return True
        return figures_supported(cv.claim, context[idx].chunk.text)


def build_verifier(config: VerificationConfig,
                   generator: LLMConfig | None = None) -> CitationVerifier | None:
    """Construct a verifier from config, or ``None`` when verification is off.

    Offline judge providers (mock/hashing) select the deterministic
    :class:`LexicalJudge`; everything else builds an :class:`LLMJudge` from the
    separately-configured judge provider (cross-vendor independence).

    ``generator`` (the answer-writing LLM config) is used only to enforce
    independence: an LLM judge that is the *same model* as the generator is not an
    independent check (a model grading its own output), so it is rejected. This
    matters most for the local stack, where the generator is ``gemma3`` and the
    judge must therefore be a *different* local model (e.g. ``mistral:7b``), never
    ``gemma3`` again.
    """
    if not config.enabled:
        return None
    provider = (config.judge.provider or "mock").lower()
    if provider in {"mock", "hashing", "offline", ""}:
        return CitationVerifier(LexicalJudge(), config)
    if generator is not None and _same_model(config.judge, generator):
        raise ValueError(
            "LLM-as-judge independence violated: the verification judge "
            f"({config.judge.provider}:{config.judge.model}) is the same model as "
            f"the generator ({generator.provider}:{generator.model}). A model cannot "
            "independently grade its own output. Configure verification.judge to a "
            "different model/vendor (e.g. a local 'mistral:7b' judge alongside a "
            "'gemma3' generator, or a cross-vendor cloud judge)."
        )
    from ..llm.factory import build_llm

    judge_llm = build_llm(config.judge)
    checker = None
    if config.qualifier_check:
        from .qualifiers import QualifierChecker

        checker = QualifierChecker(judge_llm)
    return CitationVerifier(LLMJudge(judge_llm), config, qualifier_checker=checker)


def _same_model(judge: LLMConfig, generator: LLMConfig) -> bool:
    """True if the judge and generator are the same provider+model (self-grading)."""
    return (judge.provider.lower(), judge.model) == (generator.provider.lower(), generator.model)
