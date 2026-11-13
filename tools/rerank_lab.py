#!/usr/bin/env python3
"""Cross-encoder reranking on top of the production retriever (MMR + company-blind HyDE).

With the gold passage guaranteed to be in the index (v3_gold), 20% of FinDER
questions still miss it in the top 8 (docs/EXPERIMENTS.md §7.2): a ranking
problem. This re-scores the production candidate pool (the 60 best chunks by the
mean cosine to the question and its company-blind HyDE passage, inside the ticker
filter) with a cross-encoder and compares Hit@8 / MRR@8, paired, with the
production top 8 (MMR over the same pool).

Variants (all over the same 60-chunk pool):
  prod            MMR (lambda 0.85) on the dense score — production
  rr_<m>          top 8 by reranker score
  rr_<m>_mmr      MMR with the reranker score (min-max scaled per query) as relevance
  rrf_<m>         reciprocal-rank fusion of dense and reranker ranks (k=60), top 8

Rerankers: minilm (ms-marco-MiniLM-L-6-v2), bge (bge-reranker-v2-m3),
qwen (Qwen3-Reranker-0.6B). Scores are cached in out/rerank/scores_<m>.json.

    python tools/rerank_lab.py --index data/sec_index_v3 --rerankers minilm qwen bge
    python tools/rerank_lab.py --cases eval/synth_claude.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rewrite_lab import FINDER, INSTR, K, LAM, POOL, boot_ci, cache_path, cases_for, load, mmr  # noqa: E402

OUT = Path("out/rerank")


def scorer(name: str):
    from retrieval_lab import CrossEncoderScorer, QwenReranker
    return QwenReranker() if name == "qwen" else CrossEncoderScorer(name)


def mmr_on(rel: np.ndarray, emb: np.ndarray, pool: list[int]) -> list[int]:
    """MMR over ``pool`` with relevance ``rel`` (indexed like emb)."""
    return mmr(pool, rel, emb, LAM, K)


def main() -> int:
    import chromadb
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/sec_index_v3")
    ap.add_argument("--cases", default=FINDER)
    ap.add_argument("--split", default="all")
    ap.add_argument("--model", default="gemma4:12b", help="HyDE cache (company-blind)")
    ap.add_argument("--rerankers", nargs="+", default=["minilm", "qwen", "bge"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = Path(args.index)
    blind = json.loads(cache_path(args.model, True).read_text())
    cases = [c for c in cases_for(args.split, args.cases) if c["id"] in blind]
    tickers = sorted({c["ticker"] for c in cases})
    print(f"{index} {Path(args.cases).stem}: {len(cases)} questions with HyDE", flush=True)
    chunks = load(index, set(tickers))
    col = chromadb.PersistentClient(path=str(index / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float16})

    # Pass 1: dense pools (no reranker loaded yet).
    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)
    work = []  # (case, meta, emb, s, pool, rel)
    for t in tickers:
        meta = chunks[t]
        ids = [m["id"] for m in meta]
        got = col.get(ids=ids, include=["embeddings"])
        e = dict(zip(got["ids"], got["embeddings"], strict=True))
        emb = np.asarray([e[i] for i in ids], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        for c in by_t[t]:
            rel = {i for i, m in enumerate(meta)
                   if passage_hit(m["text"], c["relevant_passages"], 0.5, CONTAINMENT_THRESHOLD)}
            if not rel:
                continue
            v = qwen.encode([INSTR + c["question"], INSTR + blind[c["id"]]["hyde"]],
                            normalize_embeddings=True)
            s = (emb @ v.T).mean(axis=1)
            pool = list(np.argsort(-s)[:POOL])
            work.append((c, meta, emb, s, pool, rel))
    del qwen
    print(f"scorable {len(work)}; gold in pool for {sum(bool(set(p) & r) for *_, p, r in work)}",
          flush=True)

    def rank_of(lst, rel):
        return next((k for k, i in enumerate(lst, 1) if i in rel), 0)

    rows = []
    for c, meta, emb, s, pool, rel in work:
        r = rank_of(mmr_on(s, emb, pool), rel)
        rows.append({"id": c["id"], "variant": "prod", "hit": 0 < r <= K, "rr": 1 / r if r else 0.0})

    # Pass 2: one reranker at a time (memory), cached scores.
    for m in args.rerankers:
        cpath = OUT / f"scores_{m}.json"
        cache = json.loads(cpath.read_text()) if cpath.exists() else {}
        model, t0, n_new = None, time.time(), 0
        for j, (c, meta, emb, _s, pool, rel) in enumerate(work, 1):
            need = [i for i in pool if f"{c['id']}|{meta[i]['id']}" not in cache]
            if need:
                model = model or scorer(m)
                got = model(c["question"], [meta[i]["text"] for i in need])
                for i, sc in zip(need, got, strict=True):
                    cache[f"{c['id']}|{meta[i]['id']}"] = float(sc)
                n_new += len(need)
            if j % 25 == 0:
                cpath.write_text(json.dumps(cache))
                print(f"  {m}: {j}/{len(work)} questions, {n_new} pairs scored "
                      f"({time.time() - t0:.0f}s)", flush=True)
            sc = np.full(len(meta), -1e9, dtype=np.float32)
            for i in pool:
                sc[i] = cache[f"{c['id']}|{meta[i]['id']}"]
            by_rr = sorted(pool, key=lambda i: -sc[i])
            lo, hi = sc[pool].min(), sc[pool].max()
            scaled = np.zeros(len(meta), dtype=np.float32)
            scaled[pool] = (sc[pool] - lo) / (hi - lo + 1e-9)
            dense_rank = {i: k for k, i in enumerate(pool, 1)}
            rr_rank = {i: k for k, i in enumerate(by_rr, 1)}
            fused = sorted(pool, key=lambda i: -(1 / (60 + dense_rank[i]) + 1 / (60 + rr_rank[i])))
            for name, lst in ((f"rr_{m}", by_rr[:K]), (f"rr_{m}_mmr", mmr_on(scaled, emb, pool)),
                              (f"rrf_{m}", fused[:K])):
                r = rank_of(lst, rel)
                rows.append({"id": c["id"], "variant": name, "hit": 0 < r <= K,
                             "rr": 1 / r if r else 0.0})
        cpath.write_text(json.dumps(cache))
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"  {m}: done, {n_new} new pairs in {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="id", columns="variant", values=["hit", "rr"])
    base = piv["hit"]["prod"].astype(float)
    res = []
    for v in df.variant.unique():
        hit = piv["hit"][v].astype(float)
        dh = (hit - base).to_numpy()
        lo, hi = boot_ci(dh)
        res.append({"variant": v, "Hit@8": hit.mean(), "MRR@8": piv["rr"][v].mean(),
                    "dHit": dh.mean(), "CI95": f"[{lo:+.3f},{hi:+.3f}]",
                    "W/L": f"{int((dh > 0).sum())}/{int((dh < 0).sum())}"})
    pd.set_option("display.width", 200)
    tab = pd.DataFrame(res).set_index("variant").sort_values("Hit@8", ascending=False).round(3)
    stem = f"{index.name}_{Path(args.cases).stem}"
    txt = (f"{index} {Path(args.cases).stem}: scorable {len(piv)}; pool = top {POOL} by "
           f"mean cos(question, company-blind HyDE); Delta vs prod (MMR lambda {LAM})\n\n"
           + tab.to_string())
    (OUT / f"{stem}.txt").write_text(txt)
    df.to_json(OUT / f"{stem}_rows.json", orient="records")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
