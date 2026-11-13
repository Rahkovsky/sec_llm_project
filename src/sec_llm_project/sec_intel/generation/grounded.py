"""Grounded answer generation (plan items 8, 10).

Pipeline: retrieve evidence -> abstain early if evidence is too weak -> prompt a
zero-temperature LLM constrained to the retrieved excerpts -> attach only the
citations the model actually relied on -> estimate confidence -> abstain again
if confidence is below the configured floor. Every answer carries its citations,
confidence, and (optionally) an audit record.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ..core.audit import AuditLogger, null_logger
from ..core.config import GenerationConfig, VerificationConfig
from ..core.tracing import CHAIN, EVALUATOR, LLM, RETRIEVER, NullTracer, Tracer
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
from .qualifiers import QualifierItem, is_candidate
from .verifier import CitationVerifier, cited_numbers

_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class GroundedAnswerer:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider,
                 config: GenerationConfig | None = None,
                 audit: AuditLogger | None = None,
                 verifier: CitationVerifier | None = None,
                 tracer: Tracer | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or GenerationConfig()
        self.audit = audit or null_logger()
        self.verifier = verifier
        self.tracer = tracer or NullTracer()

    def answer(self, question: str, *, filters: dict[str, Any] | None = None,
               top_k: int | None = None) -> Answer:
        # Every exit path (incl. all abstentions) carries the ranked retrieval list
        # and the raw first generation, so failures can be analysed after a run.
        trace: dict[str, Any] = {}
        ans = self._answer(question, filters=filters, top_k=top_k, trace=trace)
        ans.extra["retrieval"] = [_retrieval_row(r) for r in trace.get("results", [])]
        period = getattr(self.retriever, "last_period", None)
        if period:
            ans.extra["period_filter"] = period
        if "raw_generation" in trace:
            ans.extra["raw_generation"] = trace["raw_generation"]
        ans.extra.setdefault("abstain_reason", ans.abstain_reason if ans.abstained else "")
        return ans

    def _answer(self, question: str, *, filters: dict[str, Any] | None,
                top_k: int | None, trace: dict[str, Any]) -> Answer:
        cfg = self.config
        # Root span wraps the whole request; child spans (retrieve/generate/verify)
        # capture the per-stage data the pipeline already computes. With the default
        # NullTracer this is all no-ops.
        with self.tracer.span("answer", kind=CHAIN, input=question) as root:
            with self.tracer.span("retrieve", kind=RETRIEVER) as rspan:
                results = self.retriever.retrieve(question, top_k=top_k, filters=filters)
                trace["results"] = results
                context = results[: cfg.max_context_chunks]
                rspan.set(
                    n_retrieved=len(results),
                    latency_ms=round(self.retriever.last_latency_ms, 2),
                    documents=[{"id": r.chunk.chunk_id,
                                "score": r.components.get("fused"),
                                "content": r.chunk.text[:400]} for r in context])

            evidence_score = self._evidence_score(context)
            root.set(evidence_score=round(evidence_score, 4), n_retrieved=len(results))
            if not context or evidence_score < cfg.min_evidence_score:
                root.set(abstained=True, abstain_reason="insufficient_evidence")
                return self._abstain(
                    question, context,
                    f"insufficient retrieval evidence (score={evidence_score:.3f} < "
                    f"{cfg.min_evidence_score})",
                )

            messages = [
                Message("system", prompts.system_prompt(cfg.qualifier_rule, cfg.concise_rule)),
                Message("user", prompts.build_user_prompt(question, context)),
            ]
            with self.tracer.span("generate", kind=LLM) as gspan:
                gen = self.llm.generate(messages, temperature=0.0)
                gspan.set(model=gen.model, output=gen.text)
            trace["raw_generation"] = gen.text

            text, partial = strip_insufficient(gen.text)
            if text is None:
                root.set(abstained=True, abstain_reason="model_insufficient")
                return self._abstain(question, context,
                                     "model reported insufficient evidence", model=gen.model)
            gen = replace(gen, text=text)

            answer = self._assemble(question, gen, context, results, evidence_score)
            answer.extra["partial_insufficient"] = partial

            if self.verifier is not None:
                with self.tracer.span("verify", kind=EVALUATOR) as vspan:
                    answer = self._verify_and_gate(answer, context, results,
                                                   evidence_score, messages)
                    vr = answer.extra.get("verification") or {}
                    vspan.set(groundedness=vr.get("groundedness"),
                              contradicted=vr.get("contradicted"),
                              supported=vr.get("supported"), total=vr.get("total"))
                if answer.abstained:
                    root.set(abstained=True, abstain_reason=answer.abstain_reason)
                    return answer

            if len(answer.citations) < cfg.min_citations:
                root.set(abstained=True, abstain_reason="too_few_citations")
                return self._finalize_abstain(answer, "too few grounded citations")
            if answer.confidence < cfg.abstain_below_confidence:
                root.set(abstained=True, abstain_reason="low_confidence")
                return self._finalize_abstain(answer, f"low confidence ({answer.confidence:.3f})")

            audit_payload: dict[str, Any] = {
                "question": question, "model": answer.model, "confidence": answer.confidence,
                "n_citations": len(answer.citations), "abstained": False,
                "chunk_ids": [c.chunk_id for c in answer.citations],
            }
            if "verification" in answer.extra:
                audit_payload["verification"] = answer.extra["verification"]
            self.audit.log("answer", audit_payload)
            root.set(output=answer.text, confidence=round(answer.confidence, 4),
                     abstained=False, n_citations=len(answer.citations))
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

        qual: dict[str, Any] | None = None
        if self.verifier.qualifier_checker is not None:
            dropped, qual = self._dropped_qualifiers(report)
            if dropped and vcfg.qualifier_repair:
                fixed = self._repair_qualifiers(messages, answer, report, dropped, context,
                                                results, evidence_score)
                qual["repair_attempted"] = True
                if fixed is not None:
                    answer, report, qual_after = fixed
                    qual = {**qual_after, "repair_attempted": True, "repaired": True,
                            "dropped_before": qual["dropped"]}

        self._apply_verdicts(answer, report, context)
        answer.extra["verification"] = report.to_dict()
        if qual is not None:
            answer.extra["qualifiers"] = qual
        # Claim verdicts index into `context`, which is otherwise discarded —
        # persist it so provenance (claim -> evidence) survives the Answer.
        answer.extra["verification"]["evidence"] = [
            Citation.from_result(r).to_dict() for r in context]
        if report.total:
            answer.confidence = _clamp(answer.confidence * (0.4 + 0.6 * report.groundedness))

        if report.claims and not report.total:
            # every claim is "the excerpts do not contain ..." -- an abstention in prose
            return self._finalize_abstain(answer, "answer only describes missing evidence")
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
        text2, _ = strip_insufficient(gen2.text)
        if text2 is None:
            return None
        gen2 = replace(gen2, text=text2)
        answer2 = self._assemble(answer.question, gen2, context, results, evidence_score)
        report2 = self.verifier.verify(answer2, context)
        if (report2.groundedness >= report.groundedness
                and report2.contradicted <= report.contradicted):
            report2.repaired = True
            return answer2, report2
        return None

    def _dropped_qualifiers(self, report: VerificationReport
                            ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        """Supported claims that state as fact what their quote hedges."""
        assert self.verifier is not None and self.verifier.qualifier_checker is not None
        cands = [cv for cv in report.scored
                 if cv.verdict in (SUPPORTED, PARTIAL) and cv.quote
                 and is_candidate(cv.claim, cv.quote)]
        verdicts = self.verifier.qualifier_checker.check(
            [QualifierItem(cv.claim, cv.quote) for cv in cands])
        dropped = [(cv.claim, cv.quote) for cv, v in zip(cands, verdicts, strict=False) if v]
        info = {"candidates": len(cands),
                "unchecked": sum(v is None for v in verdicts),
                "dropped": [{"claim": c, "quote": q} for c, q in dropped]}
        return dropped, info

    def _repair_qualifiers(self, messages: list[Message], answer: Answer,
                           report: VerificationReport, dropped: list[tuple[str, str]],
                           context: list[RetrievalResult], results: list[RetrievalResult],
                           evidence_score: float
                           ) -> tuple[Answer, VerificationReport, dict[str, Any]] | None:
        """One re-prompt to restore the qualifiers; kept only if it is no worse."""
        assert self.verifier is not None
        gen2 = self.llm.generate([
            *messages, Message("assistant", answer.text),
            Message("user", prompts.build_qualifier_repair_prompt(dropped))],
            temperature=0.0)
        text2, _ = strip_insufficient(gen2.text)
        if text2 is None:
            return None
        answer2 = self._assemble(answer.question, replace(gen2, text=text2), context,
                                 results, evidence_score)
        report2 = self.verifier.verify(answer2, context)
        dropped2, info2 = self._dropped_qualifiers(report2)
        if (report2.groundedness >= report.groundedness
                and report2.contradicted <= report.contradicted
                and len(dropped2) < len(dropped)):
            report2.repaired = report.repaired
            return answer2, report2, info2
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
        return [k - 1 for k in cited_numbers(text) if 1 <= k <= n]

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


def strip_insufficient(text: str) -> tuple[str | None, bool]:
    """Apply the INSUFFICIENT_EVIDENCE flag.

    Returns (None, False) for an abstention: the flag with no cited content around
    it. Otherwise returns the text without the flag and whether a flag was removed
    (a partial answer: some sub-questions answered with citations, others not).
    Previously any occurrence abstained, discarding ~10 substantive v2 answers that
    flagged one sub-part (e.g. finder_0042). The cited remainder still goes through
    the verifier and the groundedness gate.
    """
    if _INSUFFICIENT not in text:
        return text, False
    rest = re.sub(r"[ \t]*\**%s\**[.:]?[ \t]*" % _INSUFFICIENT, " ", text)
    rest = re.sub(r"\n[ \t]+\n", "\n\n", rest).strip()
    if not cited_numbers(rest) or len(rest.split()) < 8:
        return None, False
    return rest, True


def _retrieval_row(r: RetrievalResult) -> dict[str, Any]:
    m = r.chunk.metadata
    return {"rank": r.rank, "chunk_id": r.chunk.chunk_id, "ticker": m.ticker,
            "filing_type": m.filing_type, "filing_date": m.filing_date,
            "item_number": r.chunk.item_number, "section_title": r.chunk.section_title,
            **r.components}
