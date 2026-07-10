#!/usr/bin/env python3
"""How much do BM25 and dense retrieval agree, on the real 260-question gold set?

For each question, ranks the top candidate_k passages independently by BM25 and
by dense (embedding) similarity — no fusion — and reports, over the chunks each
method actually returns:

* overlap    — fraction of the top-8 sets the two methods share
* tau        — Kendall's tau rank concordance, restricted to chunks both
               methods surfaced (order agreement, -1 = reversed, 1 = identical)
* rank diff  — mean |rank_bm25 - rank_dense| for those shared chunks
* shared     — how many of the candidate_k=60 candidates the two methods share
               at all (context for how meaningful tau/rank-diff are)

Also emits scatter.png: pooled (bm25_rank, dense_rank) for every shared
candidate across all 260 questions, the plot for the "how concordant are they"
slide.

    python tools/retrieval_agreement.py [--config config/local.yaml] [--out out/seminar/img]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scipy.stats import kendalltau

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.embeddings.factory import build_embedder
from sec_llm_project.sec_intel.index.store import SECIndex
from sec_llm_project.sec_intel.retrieval.bm25 import BM25Index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--out", default="out/seminar/img")
    ap.add_argument("--k", type=int, default=8, help="top-k for the overlap stat")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions (testing)")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    print("loading embedder + index (chroma + bm25.json) ...", flush=True)
    embedder = build_embedder(cfg.embedding)
    index = SECIndex.load(cfg.index.path, expected=embedder.info)
    bm25 = BM25Index.from_saved(Path(cfg.index.path, "bm25.json"))
    assert bm25 is not None, "bm25.json missing — run tools/finalize_index.py first"

    cases = json.loads(Path(args.cases).read_text())
    if args.limit:
        cases = cases[: args.limit]
    candidate_k = cfg.retrieval.candidate_k
    overlaps, taus, rankdiffs, shared_counts = [], [], [], []
    pooled_bm25_rank, pooled_dense_rank = [], []

    for i, case in enumerate(cases):
        q = case["question"]
        qvec = embedder.embed_query(q)
        dense = index.dense_search(qvec, candidate_k)
        lex = bm25.search(q, candidate_k)
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense, 1)}
        lex_rank = {cid: r for r, (cid, _) in enumerate(lex, 1)}

        top_dense = set(list(dense_rank)[: args.k])
        top_lex = set(list(lex_rank)[: args.k])
        overlaps.append(len(top_dense & top_lex) / args.k)

        shared = set(dense_rank) & set(lex_rank)
        shared_counts.append(len(shared))
        if len(shared) >= 2:
            dr = [dense_rank[c] for c in shared]
            lr = [lex_rank[c] for c in shared]
            tau, _ = kendalltau(dr, lr)
            if tau == tau:  # skip NaN (constant ranks)
                taus.append(tau)
            rankdiffs.append(sum(abs(a - b) for a, b in zip(dr, lr, strict=True)) / len(shared))
            pooled_bm25_rank.extend(lr)
            pooled_dense_rank.extend(dr)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cases)} questions", flush=True)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "n_questions": len(cases),
        "candidate_k": candidate_k,
        "top_k_for_overlap": args.k,
        "mean_top_k_overlap": round(avg(overlaps), 4),
        "mean_kendall_tau": round(avg(taus), 4),
        "mean_rank_diff": round(avg(rankdiffs), 2),
        "mean_shared_candidates": round(avg(shared_counts), 2),
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval_agreement.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=160)
    ax.scatter(pooled_bm25_rank, pooled_dense_rank, s=10, alpha=0.15, color="#3B82F6",
              edgecolors="none")
    lim = candidate_k + 1
    ax.plot([0, lim], [0, lim], "--", color="#DC2626", linewidth=1.3, label="perfect agreement")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("BM25 rank (1 = top)", fontsize=12)
    ax.set_ylabel("Embedding rank (1 = top)", fontsize=12)
    ax.set_title(f"Same passage, ranked by each method\n"
                 f"({sum(shared_counts):,} shared candidates across {len(cases)} questions)",
                 fontsize=12.5)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "rank_scatter.png", facecolor="white")
    print(f"wrote {out}/rank_scatter.png and retrieval_agreement.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
