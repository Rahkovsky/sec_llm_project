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
from typing import Protocol, runtime_checkable

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

_CITE_RE = re.compile(r"\[(\d+)\]")
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
        if not isinstance(raw, dict):
            continue
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

    def __init__(self, judge: Judge, config: VerificationConfig | None = None) -> None:
        self.judge = judge
        self.config = config or VerificationConfig()

    def verify(self, answer: Answer, context: list[RetrievalResult]) -> VerificationReport:
        claims = self.judge.judge(answer.text, context)
        for cv in claims:
            cv.valid_quote = self._quote_in_source(cv, context)
            cv.numeric_ok = self._numeric_in_source(cv, context)
            # A claim asserted as supported is downgraded if either invariant fails:
            # the supporting quote isn't actually in the cited source, or the claim
            # asserts a figure that does not appear in the cited source (a numeric
            # hallucination that surface-overlap/entailment judges tend to miss).
            if cv.verdict in (SUPPORTED, PARTIAL) and cv.citation is not None and (
                not cv.valid_quote or not cv.numeric_ok
            ):
                cv.verdict = UNSUPPORTED
        return VerificationReport(claims=claims, judge_model=self.judge.name)

    @staticmethod
    def _quote_in_source(cv: ClaimVerdict, context: list[RetrievalResult]) -> bool:
        if cv.citation is None or not cv.quote:
            return False
        idx = cv.citation - 1
        if not (0 <= idx < len(context)):
            return False
        return _normalize_ws(cv.quote) in _normalize_ws(context[idx].chunk.text)

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

    return CitationVerifier(LLMJudge(build_llm(config.judge)), config)


def _same_model(judge: LLMConfig, generator: LLMConfig) -> bool:
    """True if the judge and generator are the same provider+model (self-grading)."""
    return (judge.provider.lower(), judge.model) == (generator.provider.lower(), generator.model)
