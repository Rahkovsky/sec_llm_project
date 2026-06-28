"""Evaluation harness over a curated question set (plan item 9).

Runs retrieval and grounded generation for each case and reports Recall@k, MRR,
MAP@k, citation correctness, hallucination rate, abstention behaviour, and
latency. Gold relevance is keyed on SEC item numbers (e.g. risk questions should
retrieve Item 1A), which is robust to re-chunking.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from ..generation.verifier import CitationVerifier, LexicalJudge
from ..pipeline import SECIntelPipeline
from . import metrics

_PASSAGE_THRESHOLD = 0.5  # token-recall floor to count a chunk as relevant


@dataclass
class EvalCase:
    id: str
    question: str
    relevant_items: list[str]
    ticker: str | None = None
    answerable: bool = True
    # Passage-level gold relevance (FinDER / FinanceBench): verbatim text
    # excerpts from the filing.  When set, retrieval is scored by token-overlap
    # with these passages instead of by SEC item number.
    relevant_passages: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    k: int
    n_cases: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_case: list[dict[str, Any]] = field(default_factory=list)
    # Which judge produced the faithfulness verdicts, so the numbers are
    # self-describing (a lexical judge cannot detect contradictions and only
    # measures keyword overlap, not entailment — see metric naming below).
    judge_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "n_cases": self.n_cases, "judge_model": self.judge_model,
                "metrics": self.metrics, "per_case": self.per_case}

    def render(self) -> str:
        head = f"Evaluation over {self.n_cases} cases (k={self.k})"
        if self.judge_model:
            head += f"  [judge: {self.judge_model}]"
        lines = [head, "-" * 40]
        lines.extend(f"  {key:24s}: {self.metrics[key]:.4f}" for key in sorted(self.metrics))
        return "\n".join(lines)


class EvaluationRunner:
    def __init__(self, pipeline: SECIntelPipeline) -> None:
        self.pipeline = pipeline
        # Measure faithfulness even when production gating is off; reuse the
        # pipeline's configured judge if any, else a deterministic offline one.
        self.verifier = pipeline.verifier or CitationVerifier(LexicalJudge())

    def _faithfulness(self, answer, results: list) -> dict:
        """Groundedness for one answer, reusing ``ask``'s report if present."""
        report = answer.extra.get("verification")
        if report is None:  # verification off in config -> measure with our judge
            ctx_n = self.pipeline.config.generation.max_context_chunks
            report = self.verifier.verify(answer, results[:ctx_n]).to_dict()
        return report

    def run(self, cases: list[EvalCase], *, k: int = 8, generate: bool = True) -> EvalReport:  # noqa: PLR0912
        rankings: list[list[str]] = []
        relevants: list[list[str]] = []
        recalls: list[float] = []
        maps: list[float] = []
        latencies: list[float] = []
        cite_scores: list[float] = []
        groundedness_scores: list[float] = []
        hallucinations = 0
        contradiction_count = 0
        fully_grounded = 0
        correct_abstentions = 0
        answerable_count = 0
        per_case: list[dict[str, Any]] = []

        for case in cases:
            filters = {"ticker": case.ticker.upper()} if case.ticker else None
            results = self.pipeline.retriever.retrieve(case.question, top_k=k, filters=filters)
            passage_mode = bool(case.relevant_passages)

            # Resolve ranking keys + gold set for both relevance modes:
            #  * passage mode (FinDER / FinanceBench): key = chunk id; a chunk is
            #    relevant if its text token-overlaps a gold evidence passage.
            #  * item mode (synthetic / hand-labelled): key = SEC item number.
            # Citations are later scored against ``gold`` using the same key, so
            # citation-correctness and hallucination work in both modes.
            if passage_mode:
                ranked_keys = [r.chunk.chunk_id for r in results]
                gold: set[str] = {
                    r.chunk.chunk_id for r in results
                    if metrics.passage_hit(r.chunk.text, case.relevant_passages,
                                           _PASSAGE_THRESHOLD)
                }
            else:
                ranked_keys = []  # dedupe item numbers, preserving rank order
                for r in results:
                    if r.chunk.item_number not in ranked_keys:
                        ranked_keys.append(r.chunk.item_number)
                gold = set(case.relevant_items)

            rankings.append(ranked_keys)
            relevants.append(list(gold))
            recall = metrics.recall_at_k(ranked_keys, gold, k)
            ap = metrics.average_precision_at_k(ranked_keys, gold, k)
            rr = metrics.reciprocal_rank(ranked_keys, gold)
            recalls.append(recall)
            maps.append(ap)
            latencies.append(self.pipeline.retriever.last_latency_ms)

            row: dict[str, Any] = {
                "id": case.id, "question": case.question,
                "recall@k": round(recall, 4), "rr": round(rr, 4),
                "retrieved": ranked_keys[:k],
            }

            if generate:
                answer = self.pipeline.ask(case.question, filters=filters, top_k=k)
                cited_keys = [c.chunk_id if passage_mode else c.item_number
                              for c in answer.citations]
                row["abstained"] = answer.abstained
                row["confidence"] = round(answer.confidence, 4)
                if case.answerable:
                    answerable_count += 1
                    cc = metrics.citation_correctness(cited_keys, gold)
                    cite_scores.append(cc)
                    row["citation_correctness"] = round(cc, 4)
                    # Hallucination: produced a non-abstained answer with no
                    # citation pointing at relevant evidence.
                    if not answer.abstained and cc == 0.0:
                        hallucinations += 1
                        row["hallucinated"] = True
                    # Faithfulness: claim-level grounding of the answer given.
                    if not answer.abstained:
                        fr = self._faithfulness(answer, results)
                        g = float(fr["groundedness"])
                        groundedness_scores.append(g)
                        contradiction_count += int(fr["contradicted"] > 0)
                        fully_grounded += int(g >= 1.0)
                        row["groundedness"] = round(g, 4)
                elif answer.abstained:
                    correct_abstentions += 1
                else:
                    hallucinations += 1
                    row["hallucinated"] = True
            per_case.append(row)

        agg: dict[str, float] = {
            "recall@k": _mean(recalls),
            "mrr": metrics.mrr(rankings, [list(r) for r in relevants]),
            "map@k": _mean(maps),
            "mean_latency_ms": _mean(latencies),
            "p95_latency_ms": _percentile(latencies, 95),
        }
        if generate:
            agg["citation_correctness"] = _mean(cite_scores)
            denom = max(1, len(cases))
            agg["hallucination_rate"] = hallucinations / denom
            unanswerable = sum(1 for c in cases if not c.answerable)
            if unanswerable:
                agg["abstention_recall"] = correct_abstentions / unanswerable
            # Faithfulness aggregates over answered (non-abstained) answerable cases.
            # The metric NAME reflects what the judge can actually measure: a lexical
            # (keyword-overlap) judge cannot detect contradictions and does not test
            # entailment, so we report it as `lexical_groundedness` and omit
            # contradiction_rate (it would be a structural 0, not an observation).
            if groundedness_scores:
                scored = len(groundedness_scores)
                lexical = getattr(self.verifier.judge, "name", "") == LexicalJudge.name
                agg["lexical_groundedness" if lexical else "mean_groundedness"] = \
                    _mean(groundedness_scores)
                agg["fully_grounded_rate"] = fully_grounded / scored
                if not lexical:
                    agg["contradiction_rate"] = contradiction_count / scored
        judge_model = self.verifier.judge.name if generate else ""
        return EvalReport(k=k, n_cases=len(cases), metrics=agg, per_case=per_case,
                          judge_model=judge_model)


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])
