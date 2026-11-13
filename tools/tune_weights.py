#!/usr/bin/env python3
"""Sweep BM25 / dense fusion weights and report Hit@k + MRR.

Loads a pre-built index and eval set and runs retrieval-only sweeps across
bm25_weight values (dense_weight = 1 - bm25_weight). Dense and BM25 candidates
are computed ONCE per question and re-fused for every weight (fusion is the only
weight-dependent step), so a full sweep costs about one retrieval pass.

Relevance (passage mode) is scored under two rules, side by side:
  strict  legacy: a chunk holds >= 50% of a gold passage's tokens
  cover   strict OR the chunk lies inside a gold passage (>= 80% of its own
          tokens; see evaluation.metrics.passage_hit). Needed because half of
          FinDER's gold passages are far longer than a chunk.
"Recall@k" in older output was really a hit rate (any relevant chunk in top-k);
it is labelled Hit@k here.

No LLM calls are made — this is pure retrieval evaluation.

Usage
-----
    python tools/tune_weights.py --eval eval/finder_semantic.json --config config/local_16gb.yaml \
        --endpoints --per-question out/tune_perq.csv --ranks out/retrieval_ranks.csv
    python tools/cv_weights.py out/tune_perq.csv --metric cover
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

_PASSAGE_THRESHOLD = 0.5   # token-recall floor to mark a chunk as relevant
RULES = {"strict": None, "cover": CONTAINMENT_THRESHOLD}


def _first_rank(ids: list[str], rel: set[str]) -> int:
    """1-based rank of the first relevant id, 0 if none."""
    return next((i for i, cid in enumerate(ids, 1) if cid in rel), 0)


def main() -> int:  # noqa: PLR0912 - linear CLI flow
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="eval/finder_semantic.json",
                    help="JSON eval set with relevant_passages (FinDER / FinanceBench)")
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--k", type=int, default=8, help="top-k for Hit@k / MRR (default 8)")
    ap.add_argument("--candidate-k", type=int, default=60,
                    help="candidate pool per retriever fed to fusion (production: 60)")
    ap.add_argument("--diag-depth", type=int, default=200,
                    help="depth of the per-retriever lists used for --ranks diagnostics")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample of the eval set (0 = use all)")
    ap.add_argument("--step", type=float, default=0.1, help="weight step size (default 0.1)")
    ap.add_argument("--fusion", default="rrf", choices=["rrf", "weighted"],
                    help="fusion strategy to test (default rrf)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--endpoints", action="store_true",
                    help="also evaluate bm25_weight 0.0 (dense only) and 1.0 (BM25 only)")
    ap.add_argument("--select", default="cover", choices=list(RULES),
                    help="relevance rule used to pick the best weight (default cover)")
    ap.add_argument("--per-question", default="",
                    help="CSV of per-question hit/rr per weight and rule, for tools/cv_weights.py")
    ap.add_argument("--ranks", default="",
                    help="CSV of per-question first-relevant rank in the dense and BM25 lists")
    args = ap.parse_args()

    from sec_llm_project.sec_intel.core.config import AppConfig
    from sec_llm_project.sec_intel.evaluation.dataset import load_cases_from_json
    from sec_llm_project.sec_intel.pipeline import SECIntelPipeline

    cfg = AppConfig.load(args.config)
    cases = [c for c in load_cases_from_json(args.eval) if c.relevant_passages]
    if not cases:
        print("ERROR: eval set has no relevant_passages.", file=sys.stderr)
        return 1
    if args.sample and args.sample < len(cases):
        import random
        random.seed(args.seed)
        cases = random.sample(cases, args.sample)

    print(f"Eval set : {args.eval}  ({len(cases)} questions)")
    print(f"Config   : {args.config}")
    print(f"Fusion   : {args.fusion}   k={args.k}   candidate_k={args.candidate_k}")
    print(f"Relevance: strict (recall >= {_PASSAGE_THRESHOLD:.0%}) and cover "
          f"(+ containment >= {CONTAINMENT_THRESHOLD:.0%}); best picked by {args.select}\n")

    cfg.retrieval.fusion = args.fusion
    cfg.retrieval.use_bm25 = cfg.retrieval.use_dense = True
    pipeline = SECIntelPipeline(cfg)
    print("Loading index...", flush=True)
    pipeline.load_index()
    retr, index = pipeline.retriever, pipeline.index
    bm25 = retr._bm25_index()
    print(f"Index loaded: {len(index)} chunks\n", flush=True)

    # ---- one retrieval pass: deep candidate lists per question
    depth = max(args.diag_depth, args.candidate_k)
    per_case = []
    t0 = time.perf_counter()
    for n, case in enumerate(cases, 1):
        filters = {"ticker": case.ticker.upper()} if case.ticker else None
        dense = index.dense_search(pipeline.embedder.embed_query(case.question), depth, filters)
        allowed = retr._allowed_ids(filters) if filters else None
        lex = bm25.search(case.question, depth, allowed)
        cand = {cid for cid, _ in dense} | {cid for cid, _ in lex}
        rel = {}
        for rule, cont in RULES.items():
            rel[rule] = set()
            for cid in cand:
                ch = index.get(cid)
                if ch is not None and passage_hit(ch.text, case.relevant_passages,
                                                  _PASSAGE_THRESHOLD, cont):
                    rel[rule].add(cid)
        per_case.append((case, dense, lex, rel))
        if n % 50 == 0:
            print(f"  retrieved {n}/{len(cases)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"Retrieval pass: {time.perf_counter() - t0:.0f}s\n")

    if args.ranks:
        with open(args.ranks, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "ticker", "rule", "dense_rank", "bm25_rank", "n_relevant_in_pool"])
            for case, dense, lex, rel in per_case:
                for rule in RULES:
                    w.writerow([case.id, case.ticker, rule,
                                _first_rank([c for c, _ in dense], rel[rule]),
                                _first_rank([c for c, _ in lex], rel[rule]), len(rel[rule])])

    # ---- fuse per weight (same code path as HybridRetriever.retrieve, minus rerank)
    step_range = [round(i * args.step, 2) for i in range(1, round(1.0 / args.step))]
    weights = [0.0, *step_range, 1.0] if args.endpoints else step_range
    pq_w = None
    if args.per_question:
        pq_fh = open(args.per_question, "w", newline="")  # noqa: SIM115
        pq_w = csv.writer(pq_fh)
        pq_w.writerow(["bm25_w", "id", "rule", "hit", "rr"])

    hdr = (f"{'bm25_w':>7} {'dense_w':>7} | {'Hit@k strict':>12} {'MRR':>6} | "
           f"{'Hit@k cover':>11} {'MRR':>6}")
    print(hdr)
    print("-" * len(hdr))
    best = (-1.0, 0.0)
    for bw in weights:
        retr.config.bm25_weight, retr.config.dense_weight = bw, round(1.0 - bw, 2)
        agg = {r: [0.0, 0.0] for r in RULES}
        for case, dense, lex, rel in per_case:
            fused = retr._fuse(dict(dense[:args.candidate_k]), dict(lex[:args.candidate_k]))
            top = [c for c, _ in sorted(fused.items(), key=lambda t: t[1], reverse=True)][:args.k]
            for rule in RULES:
                fr = _first_rank(top, rel[rule])
                hit, rr = float(fr > 0), (1.0 / fr if fr else 0.0)
                agg[rule][0] += hit
                agg[rule][1] += rr
                if pq_w is not None:
                    pq_w.writerow([bw, case.id, rule, int(hit), f"{rr:.4f}"])
        n = len(per_case)
        s, c = [x / n for x in agg["strict"]], [x / n for x in agg["cover"]]
        print(f"{bw:>7.2f} {1 - bw:>7.2f} | {s[0]:>12.4f} {s[1]:>6.4f} | {c[0]:>11.4f} {c[1]:>6.4f}",
              flush=True)
        sel = s if args.select == "strict" else c
        if sel[0] + sel[1] > best[0]:
            best = (sel[0] + sel[1], bw)
    if pq_w is not None:
        pq_fh.close()

    print(f"\nBest bm25_weight by {args.select} Hit+MRR: {best[1]:.2f} "
          f"(dense_weight {1 - best[1]:.2f}, fusion {args.fusion})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
