#!/usr/bin/env python3
"""K-fold cross-validation of the BM25/dense fusion weight.

Quantifies the double-dipping in tools/tune_weights.py, which picks the weight
and reports its score on the same questions. Reads the per-question CSV that
tune_weights.py writes with --per-question, then for each fold picks the weight
that maximises mean(Recall + MRR) on the training folds and scores it on the
held-out fold. Repeated over several shuffles. No index or model needed.

    python tools/tune_weights.py --config config/local_16gb.yaml --endpoints \
        --per-question out/tune_perq.csv
    python tools/cv_weights.py out/tune_perq.csv --folds 5 --repeats 20
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import Counter, defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="per-question CSV from tune_weights.py --per-question")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metric", default="cover", choices=["strict", "cover"],
                    help="relevance rule column to use (CSV from the current tune_weights.py)")
    args = ap.parse_args()

    # score[w][qid] = (hit, rr). Accepts the current format (rule,hit,rr) and the
    # older single-rule format (recall,rr), which was the strict rule.
    score: dict[float, dict[str, tuple[float, float]]] = defaultdict(dict)
    with open(args.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("rule", "strict") != args.metric:
                continue
            hit = float(r["hit"] if "hit" in r else r["recall"])
            score[float(r["bm25_w"])][r["id"]] = (hit, float(r["rr"]))
    weights = sorted(score)
    qids = sorted(set.intersection(*(set(score[w]) for w in weights)))
    if not qids:
        print("no questions common to all weights")
        return 1

    def mean(w: float, ids: list[str], i: int) -> float:
        return sum(score[w][q][i] for q in ids) / len(ids)

    def objective(w: float, ids: list[str]) -> float:
        return mean(w, ids, 0) + mean(w, ids, 1)

    print(f"{len(qids)} questions x {len(weights)} weights\n")
    print(f"{'bm25_w':>7}  {'Recall@k':>9}  {'MRR':>7}")
    for w in weights:
        print(f"{w:>7.2f}  {mean(w, qids, 0):>9.4f}  {mean(w, qids, 1):>7.4f}")
    best_in = max(weights, key=lambda w: objective(w, qids))
    print(f"\nIn-sample best: bm25_w={best_in:.2f}  Recall={mean(best_in, qids, 0):.4f}  "
          f"MRR={mean(best_in, qids, 1):.4f}")

    rng = random.Random(args.seed)
    cv_recall, cv_mrr, picks = [], [], Counter()
    for _ in range(args.repeats):
        ids = qids[:]
        rng.shuffle(ids)
        folds = [ids[i::args.folds] for i in range(args.folds)]
        held_r = held_m = 0.0
        for k, test in enumerate(folds):
            train = [q for j, f in enumerate(folds) if j != k for q in f]
            w = max(weights, key=lambda w: objective(w, train))
            picks[w] += 1
            held_r += sum(score[w][q][0] for q in test)
            held_m += sum(score[w][q][1] for q in test)
        cv_recall.append(held_r / len(ids))
        cv_mrr.append(held_m / len(ids))

    print(f"\n{args.folds}-fold CV x {args.repeats} repeats (held-out, mean ± sd over repeats):")
    print(f"  Recall={statistics.mean(cv_recall):.4f} ± {statistics.pstdev(cv_recall):.4f}   "
          f"MRR={statistics.mean(cv_mrr):.4f} ± {statistics.pstdev(cv_mrr):.4f}")
    print(f"  optimism (in-sample - CV): Recall {mean(best_in, qids, 0) - statistics.mean(cv_recall):+.4f}"
          f"   MRR {mean(best_in, qids, 1) - statistics.mean(cv_mrr):+.4f}")
    total = sum(picks.values())
    print("  weight picked per fold: " + ", ".join(
        f"{w:.2f}:{n / total:.0%}" for w, n in sorted(picks.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
