#!/usr/bin/env python3
"""Sweep BM25 / dense fusion weights and report Recall@k + MRR.

Loads a pre-built index and eval set, runs retrieval-only sweeps across
bm25_weight values (0.1 → 0.9 in steps of 0.1, dense_weight = 1 - bm25_weight),
and prints a table so you can pick the best weights for config/local.yaml.

No LLM calls are made — this is pure retrieval evaluation.

Usage
-----
    python tools/tune_weights.py --eval eval/finder_semantic.json --config config/local.yaml
    python tools/tune_weights.py --eval eval/financebench_2023.json --config config/local.yaml --k 10
    python tools/tune_weights.py --eval eval/finder_semantic.json --sample 100 --step 0.1
"""
from __future__ import annotations

import argparse
import sys
import time

_PASSAGE_THRESHOLD = 0.5   # token-recall floor to mark a chunk as relevant


def _token_recall(chunk_text: str, passage: str) -> float:
    p = set(passage.lower().split())
    c = set(chunk_text.lower().split())
    return len(p & c) / len(p) if p else 0.0


def _relevant_ids(results, passages: list[str]) -> set[str]:
    return {r.chunk.chunk_id for r in results
            if any(_token_recall(r.chunk.text, p) >= _PASSAGE_THRESHOLD for p in passages)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="eval/finder_semantic.json",
                    help="JSON eval set (FinDER or FinanceBench format)")
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--k", type=int, default=8, help="top-k for Recall@k / MRR (default 8)")
    ap.add_argument("--candidate-k", type=int, default=60,
                    help="candidate pool per retriever; larger = fairer for both BM25 and dense")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample of the eval set (0 = use all)")
    ap.add_argument("--step", type=float, default=0.1,
                    help="weight step size (default 0.1)")
    ap.add_argument("--fusion", default="rrf", choices=["rrf", "weighted"],
                    help="fusion strategy to test (default rrf)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sec_llm_project.sec_intel.core.config import AppConfig
    from sec_llm_project.sec_intel.evaluation.dataset import load_cases_from_json
    from sec_llm_project.sec_intel.pipeline import SECIntelPipeline

    cfg = AppConfig.load(args.config)
    cases = load_cases_from_json(args.eval)

    if args.sample and args.sample < len(cases):
        import random
        random.seed(args.seed)
        cases = random.sample(cases, args.sample)

    print(f"Eval set : {args.eval}  ({len(cases)} questions)")
    print(f"Config   : {args.config}")
    print(f"Fusion   : {args.fusion}   k={args.k}   candidate_k={args.candidate_k}")
    print(f"BM25 step: {args.step}")
    print()

    # One pipeline instantiation (loads index + BM25 once)
    cfg.retrieval.candidate_k = args.candidate_k
    cfg.retrieval.fusion = args.fusion
    pipeline = SECIntelPipeline(cfg)
    print("Loading index...", flush=True)
    pipeline.load_index()
    print(f"Index loaded: {len(pipeline.index)} chunks", flush=True)
    print()

    # Determine passage vs item-number mode
    passage_mode = any(c.relevant_passages for c in cases)
    if not passage_mode and not any(c.relevant_items for c in cases):
        print("ERROR: eval set has neither relevant_passages nor relevant_items.", file=sys.stderr)
        return 1
    print(f"Relevance mode: {f'passage-overlap (token-recall ≥ {_PASSAGE_THRESHOLD:.0%})' if passage_mode else 'item-number'}")
    print()

    bm25_range = [round(v * args.step, 10) for v in
                  range(round(0.1 / args.step), round(0.9 / args.step) + 1)]
    bm25_range = [round(v, 2) for v in
                  [i * args.step for i in range(1, round(1.0 / args.step))]]

    header = f"{'bm25_w':>7}  {'dense_w':>7}  {'Recall@k':>9}  {'MRR':>7}  {'hits':>6}  {'time_s':>7}"
    print(header)
    print("-" * len(header))

    best = {"recall": -1.0, "mrr": -1.0, "bm25_w": 0.5}

    for bm25_w in bm25_range:
        dense_w = round(1.0 - bm25_w, 2)
        pipeline.retriever.config.bm25_weight = bm25_w
        pipeline.retriever.config.dense_weight = dense_w
        pipeline.retriever.config.use_bm25 = True
        pipeline.retriever.config.use_dense = True

        recalls, rrs, hits = [], [], 0
        t0 = time.perf_counter()

        for case in cases:
            filters = {"ticker": case.ticker.upper()} if case.ticker else None
            results = pipeline.retriever.retrieve(case.question,
                                                  top_k=args.k, filters=filters)
            if passage_mode:
                rel = _relevant_ids(results, case.relevant_passages)
                ids = [r.chunk.chunk_id for r in results]
            else:
                rel = set(case.relevant_items)
                ids = []
                seen: set[str] = set()
                for r in results:
                    it = r.chunk.item_number
                    if it not in seen:
                        ids.append(it)
                        seen.add(it)

            # Recall@k
            found = set(ids[:args.k]) & rel
            rec = len(found) / len(rel) if rel else 0.0
            recalls.append(rec)
            if found:
                hits += 1

            # MRR
            rr = 0.0
            for rank, cid in enumerate(ids[:args.k], 1):
                if cid in rel:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)

        elapsed = time.perf_counter() - t0
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        mean_mrr = sum(rrs) / len(rrs) if rrs else 0.0

        print(f"{bm25_w:>7.2f}  {dense_w:>7.2f}  {mean_recall:>9.4f}  {mean_mrr:>7.4f}  "
              f"{hits:>4}/{len(cases)}  {elapsed:>6.1f}s", flush=True)  # flush: rows are
        # otherwise block-buffered when redirected to a file (~220s/row), making a
        # live run look hung on "Loading index..." until it finishes.

        if mean_recall + mean_mrr > best["recall"] + best["mrr"]:
            best = {"recall": mean_recall, "mrr": mean_mrr, "bm25_w": bm25_w}

    print()
    print(f"Best bm25_weight: {best['bm25_w']:.2f}  "
          f"(Recall@{args.k}={best['recall']:.4f}, MRR={best['mrr']:.4f})")
    print()
    print("To apply, add to config/local.yaml:")
    print("  retrieval:")
    print(f"    bm25_weight: {best['bm25_w']:.2f}")
    print(f"    dense_weight: {round(1.0 - best['bm25_w'], 2):.2f}")
    print(f"    fusion: {args.fusion}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
