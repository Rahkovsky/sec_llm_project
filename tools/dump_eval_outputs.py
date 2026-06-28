#!/usr/bin/env python3
"""Dump grounded answers as a human-gradeable sheet (judge-validation input).

LLM-as-judge / lexical-judge faithfulness scores are themselves unvalidated until
compared against human judgement. This tool runs the real pipeline over a sample
of an eval set and writes each answer with its citations and the gold evidence,
plus *blank* human-score fields, to:

* ``<out>.jsonl`` — one record per question, re-importable for agreement analysis.
* ``<out>.md``    — the same content rendered for a human to read and score.

A human fills the score fields (rubric in docs/HUMAN_EVAL.md); those scores become
the ground truth to measure how well the automated judge agrees (recommendation:
"the judge is never validated").

Usage
-----
    python tools/dump_eval_outputs.py --eval eval/financebench_2023.json --load-index
    python tools/dump_eval_outputs.py --eval eval/finder_semantic.json --load-index --sample 25 --out out/human_eval_finder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Blank human-score template (see docs/HUMAN_EVAL.md for the rubric). Kept here so
# the JSONL and the Markdown sheet stay in lock-step.
SCORE_TEMPLATE = {
    "faithfulness_0_2": None,     # grounded in the CITED evidence?
    "correctness_0_2": None,      # actually right (vs gold answer / domain truth)?
    "citation_quality_0_2": None, # citations precise and sufficient?
    "richness_0_2": None,         # complete & specific, not a cheap thin answer?
    "abstention_appropriate": None,  # true/false/"n/a" — was abstaining the right call?
    "notes": "",
}


def _truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True, help="JSON eval set (FinDER / FinanceBench format)")
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--load-index", action="store_true",
                    help="query the persisted index (otherwise builds the synthetic corpus)")
    ap.add_argument("--sample", type=int, default=15, help="number of questions to dump")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="out/human_eval", help="output path stem (.jsonl + .md)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sec_llm_project.sec_intel.core.config import AppConfig
    from sec_llm_project.sec_intel.evaluation.dataset import (
        index_sample_corpus,
        load_cases_from_json,
    )
    from sec_llm_project.sec_intel.pipeline import SECIntelPipeline

    cfg = AppConfig.load(args.config)
    pipeline = SECIntelPipeline(cfg)
    if args.load_index:
        pipeline.load_index()
    else:
        index_sample_corpus(pipeline)

    cases = load_cases_from_json(args.eval)
    if args.sample and args.sample < len(cases):
        import random
        random.seed(args.seed)
        cases = random.sample(cases, args.sample)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    records = []
    md = ["# Human evaluation sheet",
          f"_Source: {args.eval} · model: {cfg.llm.model} · {len(cases)} questions._",
          "",
          "Score each answer per the rubric in `docs/HUMAN_EVAL.md` "
          "(faithfulness / correctness / citation quality / richness, each 0-2).",
          ""]

    for i, case in enumerate(cases, 1):
        filters = {"ticker": case.ticker.upper()} if case.ticker else None
        ans = pipeline.ask(case.question, filters=filters, top_k=args.top_k)
        gold_answer = getattr(case, "gold_answer", "") or ""
        cites = [{
            "n": j + 1, "ticker": c.ticker, "filing_type": c.filing_type,
            "item": c.item_number, "quote": _truncate(c.quote, 300),
        } for j, c in enumerate(ans.citations)]
        rec = {
            "id": case.id, "question": case.question, "ticker": case.ticker,
            "gold_answer": gold_answer,
            "gold_passages": [_truncate(p, 400) for p in case.relevant_passages[:2]],
            "model_answer": ans.text,
            "abstained": ans.abstained, "confidence": round(ans.confidence, 3),
            "citations": cites,
            "scores": dict(SCORE_TEMPLATE),
        }
        records.append(rec)
        print(f"[{i}/{len(cases)}] {case.id} {'(abstained)' if ans.abstained else ''}",
              file=sys.stderr, flush=True)

        md += [f"## {i}. [{case.ticker}] {case.question}", ""]
        if gold_answer:
            md.append(f"**Gold answer:** {_truncate(gold_answer, 500)}\n")
        for p in rec["gold_passages"]:
            md.append(f"> **Gold evidence:** {p}\n")
        md += [f"**Model answer** (confidence {rec['confidence']}, "
               f"{'ABSTAINED' if ans.abstained else 'answered'}):", "", ans.text, ""]
        if cites:
            md.append("**Citations:**")
            md += [f"- [{c['n']}] {c['ticker']} {c['filing_type']} {c['item']} — \"{c['quote']}\""
                   for c in cites]
        md += ["", "**Scores** — faithfulness:_ /2 · correctness:_ /2 · "
               "citation:_ /2 · richness:_ /2 · abstention_ok:_ · notes:____",
               "", "---", ""]

    jsonl_path = out_stem.with_suffix(".jsonl")
    md_path = out_stem.with_suffix(".md")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {len(records)} records -> {jsonl_path}\n"
          f"Human-readable sheet     -> {md_path}\n"
          f"Fill the score fields, then compare against the automated judge "
          f"(see docs/HUMAN_EVAL.md).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
