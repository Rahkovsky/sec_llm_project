"""Grounded answer generation (plan items 8, 10).

Pipeline: retrieve evidence -> abstain early if evidence is too weak -> prompt a
zero-temperature LLM constrained to the retrieved excerpts -> attach only the
citations the model actually relied on -> estimate confidence -> abstain again
if confidence is below the configured floor. Every answer carries its citations,
confidence, and (optionally) an audit record.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.audit import AuditLogger, null_logger
from ..core.config import GenerationConfig, VerificationConfig
from ..core.types import (
    CONTRADICTED,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    Answer,
    Citation,
    ClaimVerdict,
    RetrievalResult,
    VerificationReport,
)
from ..llm.base import GenerationResult, LLMProvider, Message
from ..retrieval.hybrid import HybridRetriever
from . import prompts
from .verifier import CitationVerifier

_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_CITE_RE = re.compile(r"\[(\d+)\]")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class GroundedAnswerer:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider,
                 config: GenerationConfig | None = None,
                 audit: AuditLogger | None = None,
                 verifier: CitationVerifier | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or GenerationConfig()
        self.audit = audit or null_logger()
        self.verifier = verifier

    def answer(self, question: str, *, filters: dict[str, Any] | None = None,
               top_k: int | None = None) -> Answer:
        cfg = self.config
        results = self.retriever.retrieve(question, top_k=top_k, filters=filters)
        context = results[: cfg.max_context_chunks]

        evidence_score = self._evidence_score(context)
        if not context or evidence_score < cfg.min_evidence_score:
            return self._abstain(
                question, context,
                f"insufficient retrieval evidence (score={evidence_score:.3f} < "
                f"{cfg.min_evidence_score})",
            )

        messages = [
            Message("system", prompts.SYSTEM_PROMPT),
            Message("user", prompts.build_user_prompt(question, context)),
        ]
        gen = self.llm.generate(messages, temperature=0.0)

        if _INSUFFICIENT in gen.text:
            return self._abstain(question, context,
                                 "model reported insufficient evidence", model=gen.model)

        answer = self._assemble(question, gen, context, results, evidence_score)

        if self.verifier is not None:
            answer = self._verify_and_gate(answer, context, results, evidence_score, messages)
            if answer.abstained:
                return answer

        if len(answer.citations) < cfg.min_citations:
            return self._finalize_abstain(answer, "too few grounded citations")
        if answer.confidence < cfg.abstain_below_confidence:
            return self._finalize_abstain(answer, f"low confidence ({answer.confidence:.3f})")

        audit_payload: dict[str, Any] = {
            "question": question, "model": answer.model, "confidence": answer.confidence,
            "n_citations": len(answer.citations), "abstained": False,
            "chunk_ids": [c.chunk_id for c in answer.citations],
        }
        if "verification" in answer.extra:
            audit_payload["verification"] = answer.extra["verification"]
        self.audit.log("answer", audit_payload)
        return answer

    def _assemble(self, question: str, gen: GenerationResult,
                  context: list[RetrievalResult], results: list[RetrievalResult],
                  evidence_score: float) -> Answer:
        """Build an Answer (citations + confidence) from a generation result."""
        cfg = self.config
        cited_idx = self._cited_indices(gen.text, len(context))
        used = [context[i] for i in cited_idx] if cited_idx else context[: cfg.min_citations]
        citations = [Citation.from_result(r, quote_chars=cfg.quote_chars) for r in used]
        return Answer(
            question=question, text=gen.text.strip(), citations=citations,
            confidence=self._confidence(context, len(citations), evidence_score),
            model=gen.model,
            extra={"evidence_score": round(evidence_score, 4),
                   "retrieved": len(results),
                   "latency_ms": round(self.retriever.last_latency_ms, 2)},
        )

    # -------------------------------------------------------- verification
    def _verify_and_gate(self, answer: Answer, context: list[RetrievalResult],
                         results: list[RetrievalResult], evidence_score: float,
                         messages: list[Message]) -> Answer:
        assert self.verifier is not None
        vcfg = self.verifier.config
        report = self.verifier.verify(answer, context)

        if report.total and self._fails_gate(report, vcfg) and vcfg.repair:
            repaired = self._attempt_repair(messages, answer, report, context, results,
                                            evidence_score)
            if repaired is not None:
                answer, report = repaired

        self._apply_verdicts(answer, report, context)
        answer.extra["verification"] = report.to_dict()
        if report.total:
            answer.confidence = _clamp(answer.confidence * (0.4 + 0.6 * report.groundedness))

        if report.total and self._fails_gate(report, vcfg):
            reason = ("contradicted by cited evidence" if report.contradicted
                      else f"insufficient grounding (groundedness={report.groundedness:.2f})")
            self.audit.log("answer", {"question": answer.question, "abstained": True,
                                      "reason": reason, "verification": report.to_dict()})
            return self._finalize_abstain(answer, reason)
        return answer

    @staticmethod
    def _fails_gate(report: VerificationReport, vcfg: VerificationConfig) -> bool:
        return bool(
            (vcfg.abstain_on_contradiction and report.contradicted > 0)
            or report.groundedness < vcfg.groundedness_floor
        )

    def _attempt_repair(self, messages: list[Message], answer: Answer,
                        report: VerificationReport, context: list[RetrievalResult],
                        results: list[RetrievalResult],
                        evidence_score: float) -> tuple[Answer, VerificationReport] | None:
        """One-shot re-prompt dropping flagged claims; keep it only if it helps."""
        assert self.verifier is not None
        flagged = [c.claim for c in report.claims if c.verdict in (UNSUPPORTED, CONTRADICTED)]
        if not flagged:
            return None
        repair_messages = [
            *messages,
            Message("assistant", answer.text),
            Message("user", prompts.build_repair_prompt(answer.text, flagged)),
        ]
        gen2 = self.llm.generate(repair_messages, temperature=0.0)
        if _INSUFFICIENT in gen2.text:
            return None
        answer2 = self._assemble(answer.question, gen2, context, results, evidence_score)
        report2 = self.verifier.verify(answer2, context)
        if (report2.groundedness >= report.groundedness
                and report2.contradicted <= report.contradicted):
            report2.repaired = True
            return answer2, report2
        return None

    @staticmethod
    def _apply_verdicts(answer: Answer, report: VerificationReport,
                        context: list[RetrievalResult]) -> None:
        """Attach each citation's best verdict + verbatim span (matched by chunk id)."""
        num_to_chunk = {i + 1: r.chunk.chunk_id for i, r in enumerate(context)}
        rank = {SUPPORTED: 3, PARTIAL: 2, CONTRADICTED: 1, UNSUPPORTED: 0}
        best: dict[str, ClaimVerdict] = {}
        for cv in report.claims:
            if cv.citation is None:
                continue
            cid = num_to_chunk.get(cv.citation)
            if cid is None:
                continue
            if cid not in best or rank.get(cv.verdict, 0) > rank.get(best[cid].verdict, 0):
                best[cid] = cv
        for cit in answer.citations:
            cv = best.get(cit.chunk_id)
            if cv is not None:
                cit.verdict = cv.verdict
                cit.support_quote = cv.quote

    # ------------------------------------------------------------- scoring
    @staticmethod
    def _evidence_score(results: list[RetrievalResult]) -> float:
        """Strength of the best evidence, on a cosine-like [0,1]-ish scale."""
        if not results:
            return 0.0
        best = 0.0
        bm_max = max((r.components.get("bm25", 0.0) for r in results), default=0.0)
        for r in results:
            dense = r.components.get("dense", 0.0)
            bm = r.components.get("bm25", 0.0)
            norm_bm = (bm / bm_max) if bm_max > 0 else 0.0
            best = max(best, dense, norm_bm)
        return best

    @staticmethod
    def _confidence(results: list[RetrievalResult], n_citations: int,
                    evidence_score: float) -> float:
        agree = sum(
            1 for r in results
            if r.components.get("dense", 0.0) > 0 and r.components.get("bm25", 0.0) > 0
        )
        agreement = agree / len(results) if results else 0.0
        citation_factor = min(1.0, n_citations / 2.0)
        return _clamp(0.5 * evidence_score + 0.3 * citation_factor + 0.2 * agreement)

    @staticmethod
    def _cited_indices(text: str, n: int) -> list[int]:
        seen: list[int] = []
        for m in _CITE_RE.finditer(text):
            idx = int(m.group(1)) - 1
            if 0 <= idx < n and idx not in seen:
                seen.append(idx)
        return seen

    # ----------------------------------------------------------- abstention
    def _abstain(self, question: str, results: list[RetrievalResult], reason: str,
                 model: str = "") -> Answer:
        citations = [Citation.from_result(r) for r in results[: self.config.min_citations]]
        self.audit.log("answer", {"question": question, "abstained": True, "reason": reason})
        return Answer(
            question=question,
            text="I don't have enough grounded evidence in the indexed filings to answer this.",
            citations=citations, confidence=0.0, abstained=True,
            abstain_reason=reason, model=model,
        )

    def _finalize_abstain(self, answer: Answer, reason: str) -> Answer:
        answer.abstained = True
        answer.abstain_reason = reason
        answer.text = (
            "I don't have enough grounded evidence in the indexed filings to answer this confidently."
        )
        self.audit.log("answer", {"question": answer.question, "abstained": True, "reason": reason})
        return answer
