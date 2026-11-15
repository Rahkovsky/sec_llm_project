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


# segment boundaries inside a judge's quote: "..." / "…", or a sentence end ("... structure. Southwest ...")
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\s*\.\s*\.|\u2026)\s*|(?<=[a-z0-9)\]][.;])\s+(?=[A-Z])")
MIN_NEAR_QUOTE_WORDS = 8      # near matching only for quotes this long
NEAR_QUOTE_COVERAGE = 0.9     # share of the quote's words found, in order
NEAR_QUOTE_SLACK = 15         # extra source words allowed inside the matched span


def _segment_matches(quote: str, source: str, *, near: bool) -> bool:
    if _normalize_ws(quote) in _normalize_ws(source):
        return True
    q = _qwords(quote)
    if len(q.split()) >= MIN_FUZZY_QUOTE_WORDS and q in _qwords(source):
        return True
    return near and _near_match(q.split(), _qwords(source).split())


def _near_match(qw: list[str], sw: list[str]) -> bool:
    """>= 90% of the quote's words occur in order within one region of the source.

    Covers a one-word paraphrase ("enforces" for "enforce") and a page header or
    footnote marker that the chunker left inside the quoted span ("... and
    Subsidiaries Notes to Consolidated Financial Statements (Unaudited) ...")."""
    if len(qw) < MIN_NEAR_QUOTE_WORDS:
        return False
    from difflib import SequenceMatcher

    # Anchor on the longest common run, then align the whole quote inside a window
    # around it, so a phrase repeated earlier in the source cannot stretch the span.
    anchor = SequenceMatcher(None, qw, sw, autojunk=False).find_longest_match(
        0, len(qw), 0, len(sw))
    if anchor.size < 2:
        return False
    lo = max(0, anchor.b - anchor.a - NEAR_QUOTE_SLACK)
    sw = sw[lo: anchor.b + (len(qw) - anchor.a) + NEAR_QUOTE_SLACK]
    blocks = [b for b in SequenceMatcher(None, qw, sw, autojunk=False).get_matching_blocks()
              if b.size >= 2]
    if not blocks:
        return False
    matched = sum(b.size for b in blocks)
    span = blocks[-1].b + blocks[-1].size - blocks[0].b
    if matched < NEAR_QUOTE_COVERAGE * len(qw) or span > len(qw) + NEAR_QUOTE_SLACK:
        return False
    # The words that differ must not change the meaning: no negation, figure or hedge
    # may be missing from either side ("may not be sufficient" vs "is not sufficient").
    q_hit = {i for b in blocks for i in range(b.a, b.a + b.size)}
    s_hit = {i for b in blocks for i in range(b.b, b.b + b.size)}
    # the source region aligned with the whole quote, including unmatched edge words
    # ("we [believe] that ...": the dropped hedge sits before the first matched run)
    first, last = blocks[0], blocks[-1]
    s_lo = max(0, first.b - first.a)
    s_hi = min(len(sw), last.b + last.size + (len(qw) - last.a - last.size))
    diff = [w for i, w in enumerate(qw) if i not in q_hit] + [
        sw[i] for i in range(s_lo, s_hi) if i not in s_hit]
    return not any(w in _SENSITIVE_WORDS or any(ch.isdigit() for ch in w) for w in diff)


_SENSITIVE_WORDS = {
    "not", "no", "never", "nor", "without", "cannot", "unable", "none", "neither",
    "may", "might", "could", "would", "should", "believe", "believes", "expect", "expects",
    "expected", "anticipate", "anticipates", "intend", "intends", "potential", "potentially",
    "likely", "unlikely", "approximately", "estimated", "only", "less", "more", "million",
    "billion", "percent"}


def quote_matches(quote: str, source: str, *, near: bool = True) -> bool:
    """A judge's quote occurs in the source.

    Exact (whitespace-normalized) always counts; a quote of >= 5 words counts when
    its word sequence occurs contiguously in the source, ignoring punctuation, case,
    curly quotes and dashes (57 of 126 v2 'unsupported' claims had a correct quote
    rejected for a comma or a curly quote). With ``near`` (default), a quote that
    joins spans with "..." (or joins sentences) matches when every segment of 3+ words does, and a quote
    of 8+ words matches when 90% of its words occur in order in one region (36 of 44
    v5 'unsupported' claims were SUPPORTED by the judge with such a quote;
    EXPERIMENTS §20.3)."""
    if not quote.strip():
        return False
    if _segment_matches(quote, source, near=False):
        return True
    if not near:
        return False
    segments = [seg for seg in _ELLIPSIS_RE.split(quote) if len(_qwords(seg).split()) >= 3]
    if len(segments) > 1:
        return all(_segment_matches(seg, source, near=True) for seg in segments)
    return _segment_matches(quote, source, near=True)


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


class SentenceLLMJudge:
    """Compact LLM judge: one verdict per answer sentence (judge_format: sentences).

    The answer is split into sentences in code (generation/prune.py), so the judge does
    not write the claims back; each ClaimVerdict's ``claim`` is the sentence without its
    [n] markers, which are kept in ``sources`` for the figure check.
    """

    def __init__(self, llm: LLMProvider, *, quotes: bool = True) -> None:
        self.llm = llm
        self.name = str(getattr(llm, "model", None) or getattr(llm, "name", "llm-judge"))
        # quotes=False (judge_format: verdicts): the judge returns no quote; the cited
        # passage's best-matching sentence is used instead.
        self.quotes = quotes

    def judge(self, answer_text: str, context: list[RetrievalResult]) -> list[ClaimVerdict]:
        from .prune import is_lead_in, split_sentences

        sentences = split_sentences(answer_text)
        if not sentences:
            return []
        system, schema = ((prompts.SENTENCE_JUDGE_SYSTEM_PROMPT, prompts.SENTENCE_JUDGE_SCHEMA)
                          if self.quotes else
                          (prompts.VERDICT_JUDGE_SYSTEM_PROMPT, prompts.VERDICT_JUDGE_SCHEMA))
        messages = [
            Message("system", system),
            Message("user", prompts.build_sentence_judge_prompt(sentences, context)),
        ]
        result = self.llm.generate(messages, temperature=0.0, json_schema=schema)
        out = parse_sentence_verdicts(result.text, sentences, n_evidence=len(context))
        for cv, sent in zip(out, sentences, strict=True):
            if is_lead_in(sent):
                cv.verdict, cv.citation, cv.quote = SUPPORTED, None, ""
                cv.lead_in = True
        if not self.quotes:
            for cv in out:
                if cv.citation is not None and cv.verdict != UNSUPPORTED:
                    cv.quote = best_matching_sentence(cv.claim,
                                                      context[cv.citation - 1].chunk.text)
        return out


def best_matching_sentence(claim: str, passage: str) -> str:
    """The passage sentence sharing the most content words with ``claim``."""
    cands = [s for s in _sentences(passage) if s.strip()] or [passage]
    kw = _keywords(claim)
    return max(cands, key=lambda s: len(kw & _keywords(s))).strip()


_MARKER_RE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_sentence_verdicts(text: str, sentences: list[str], *,
                            n_evidence: int) -> list[ClaimVerdict]:
    """One ClaimVerdict per sentence; a sentence the judge skipped is UNSUPPORTED."""
    by_s: dict[int, dict] = {}
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            data = {}
        for raw in data.get("sentences", []) if isinstance(data, dict) else []:
            if isinstance(raw, dict) and isinstance(raw.get("s"), int):
                by_s.setdefault(raw["s"], raw)
    out: list[ClaimVerdict] = []
    for i, sent in enumerate(sentences, 1):
        raw = by_s.get(i, {})
        verdict = str(raw.get("verdict", "")).upper()
        if verdict not in _VERDICTS:
            verdict = UNSUPPORTED
        citation = raw.get("citation")
        if not (isinstance(citation, int) and 1 <= citation <= n_evidence):
            citation = None
        sources = [k for k in cited_numbers(sent) if 1 <= k <= n_evidence]
        out.append(ClaimVerdict(
            claim=_MARKER_RE.sub("", sent).strip(), verdict=verdict, citation=citation,
            quote=str(raw.get("quote", "")).strip(), sources=sources))
    return out


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


# Direction words and their opposites. A claim that says "decrease" where its own quote
# says "increase" (and never "decrease") reverses the source; the judge often passes it
# (EXPERIMENTS §20.9).
_DIRECTION_PAIRS = [
    ("increase", "decrease"), ("increased", "decreased"), ("increases", "decreases"),
    ("increasing", "decreasing"), ("higher", "lower"), ("rose", "fell"), ("grew", "declined"),
    ("more", "less"), ("above", "below"), ("gain", "loss"), ("gains", "losses"),
    ("favorable", "unfavorable"), ("positive", "negative"), ("improve", "worsen"),
    ("improved", "worsened"), ("efficiently", "inefficiently"), ("probable", "improbable"),
]
_OPPOSITE = dict(_DIRECTION_PAIRS) | {b: a for a, b in _DIRECTION_PAIRS}
_WORD_RE = re.compile(r"[a-z]+")


def direction_flip(claim: str, quote: str) -> str | None:
    """The claim's direction word whose opposite is in the quote while the word itself is
    not ("decrease/increase"), or None. Without a quote there is nothing to compare."""
    if not quote:
        return None
    cw, qw = set(_WORD_RE.findall(claim.lower())), set(_WORD_RE.findall(quote.lower()))
    for w in sorted(cw):
        if w in _OPPOSITE and _OPPOSITE[w] in qw and w not in qw:
            return f"{w}/{_OPPOSITE[w]}"
    return None


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
            cv.judge_verdict = cv.verdict  # before the quote / figure invariants
            cv.meta = cv.lead_in or is_meta_claim(cv.claim)
            self._repoint_citation(cv, context)
            cv.valid_quote = self._quote_in_source(cv, context)
            cv.numeric_ok = self._numeric_in_source(cv, context)
            if self.config.direction_check and cv.verdict in (SUPPORTED, PARTIAL):
                cv.direction_ok = direction_flip(cv.claim, cv.quote) is None
            # A claim asserted as supported is downgraded if either invariant fails:
            # the supporting quote isn't actually in the cited source, or the claim
            # asserts a figure that does not appear in the cited source (a numeric
            # hallucination that surface-overlap/entailment judges tend to miss).
            # Support must name its source: SUPPORTED/PARTIAL requires a citation
            # whose quote/figures actually check out, else it is unverifiable.
            if cv.verdict in (SUPPORTED, PARTIAL) and (
                cv.citation is None or not cv.valid_quote or not cv.numeric_ok
                or not cv.direction_ok
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
        # A sentence may combine figures from every excerpt it cites ([1, 4]).
        texts = [context[idx].chunk.text] + [
            context[k - 1].chunk.text for k in cv.sources
            if 1 <= k <= len(context) and k - 1 != idx]
        return figures_supported(cv.claim, "\n".join(texts))


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
    judge: Judge
    if config.judge_format in ("sentences", "verdicts"):
        judge = SentenceLLMJudge(judge_llm, quotes=config.judge_format == "sentences")
    else:
        judge = LLMJudge(judge_llm)
    return CitationVerifier(judge, config, qualifier_checker=checker)


def _same_model(judge: LLMConfig, generator: LLMConfig) -> bool:
    """True if the judge and generator are the same provider+model (self-grading)."""
    return (judge.provider.lower(), judge.model) == (generator.provider.lower(), generator.model)
