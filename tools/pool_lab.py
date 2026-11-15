#!/usr/bin/env python3
"""Q3-3 (docs/NEXT_STEPS.md): MMR pool size and lambda on the production retriever.

§6 chose MMR lambda 0.85 over a 60-chunk pool before HyDE and on an older index. This sweeps
both on the production scoring (mean cosine of the question and its company-blind HyDE
passage, inside the ticker filter, v4 index): pool 30 to 200, lambda 0.75 to 1.0 (1.0 = plain top
8). Paired against production (pool 60, lambda 0.85) with bootstrap CIs. Choose on dev,
confirm once on test.

    python tools/pool_lab.py --index data/sec_index_v4 --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from rewrite_lab import FINDER, INSTR, K, boot_ci, cache_path, cases_for, load, mmr

OUT = Path("out/pool")


def main() -> int:
    import chromadb
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default="data/sec_index_v4")
    ap.add_argument("--cases", default=FINDER)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="gemma4:12b", help="HyDE cache (company-blind)")
    ap.add_argument("--pools", nargs="+", type=int, default=[30, 60, 90, 120, 200])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = Path(args.index)
    blind = json.loads(cache_path(args.model, True).read_text())
    cases = [c for c in cases_for(args.split, args.cases) if c["id"] in blind]
    tickers = sorted({c["ticker"] for c in cases})
    chunks = load(index, set(tickers))
    col = chromadb.PersistentClient(path=str(index / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float16})

    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)
    rows, gold_rank = [], []
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
            order = list(np.argsort(-s))
            gold_rank.append(next(k for k, i in enumerate(order, 1) if i in rel))
            for pool in args.pools:
                for lam in args.lams:
                    lst = order[:K] if lam >= 1.0 else mmr(order[:pool], s, emb, lam, K)
                    r = next((k for k, i in enumerate(lst, 1) if i in rel), 0)
                    rows.append({"id": c["id"], "variant": f"pool{pool}_lam{lam:.2f}",
                                 "hit": 0 < r <= K, "rr": 1 / r if r else 0.0})

    gr = np.asarray(gold_rank)
    bands = [(1, 8), (9, 30), (31, 60), (61, 120), (121, 200), (201, 10**9)]
    dist = ", ".join(f"{a}-{b if b < 10**9 else '∞'}: {int(((gr >= a) & (gr <= b)).sum())}"
                     for a, b in bands)
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="id", columns="variant", values=["hit", "rr"])
    base = piv["hit"]["pool60_lam0.85"].astype(float)
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
    txt = (f"{index} {Path(args.cases).stem} split={args.split}: scorable {len(piv)}; "
           f"dense rank of the best gold chunk: {dist}\nDelta vs prod (pool 60, lambda 0.85); "
           f"lambda 1.00 = plain top {K}\n\n" + tab.to_string())
    stem = f"{index.name}_{Path(args.cases).stem}_{args.split}"
    (OUT / f"{stem}.txt").write_text(txt)
    df.to_json(OUT / f"{stem}_rows.json", orient="records")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
