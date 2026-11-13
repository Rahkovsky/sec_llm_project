#!/usr/bin/env python3
"""Dense vs BM25 vs hybrid (weighted RRF) on the ab_diagnose query sets.

Same 40 tickers / chunks / relevance as tools/ab_diagnose.py (Claude-written
synthetic questions with evidence-based relevance, FinDER dev notes and their
question rewrites). Dense lists come from cached Nomic vectors or the stored
Qwen vectors (fp32 queries); BM25 is the production index (bm25.json) with the
ticker filter; hybrid = weighted reciprocal-rank fusion (k=60) of the top-60 of
each, as in retrieval/hybrid.py.

    python tools/ab_hybrid.py      # -> out/ab_diag/hybrid.txt, hybrid_rows.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import ab_diagnose as d  # noqa: E402

PROMPTS = {
    "qwen_none": "",
    "qwen_instr1": d.QWEN_PROMPTS["qwen_instr1"],
    "qwen_instr0": d.QWEN_PROMPTS["qwen_instr0"],
    # Candidate production prompt: covers questions and terse phrases, and avoids
    # "SEC filings", which drew filing boilerplate into the top results.
    "qwen_qa": "Instruct: Given a question or search phrase about a company, retrieve the "
               "passage from its annual or quarterly report that answers it\nQuery: ",
}
BM25_W = [0.0, 0.3, 0.5, 0.7]  # 0.0 = dense only
POOL, K = 60, 8


def rrf(lists: list[list[int]], weights: list[float], k: int = 60) -> list[int]:
    score: dict[int, float] = defaultdict(float)
    for lst, w in zip(lists, weights, strict=True):
        for r, i in enumerate(lst, 1):
            score[i] += w / (k + r)
    return sorted(score, key=lambda i: -score[i])


def main() -> int:
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit
    from sec_llm_project.sec_intel.retrieval.bm25 import BM25Index

    qs = d.query_sets()
    nq = json.loads((d.OUT / "nomic_q.json").read_text())
    tickers = sorted({x["ticker"] for x in qs})
    chunks = d.load_chunks(set(tickers))
    col = chromadb.PersistentClient(path=str(d.INDEX / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float32})
    print("loading BM25...", flush=True)
    bm25 = BM25Index.from_saved(d.INDEX / "bm25.json")

    by_t = defaultdict(list)
    for x in qs:
        by_t[x["ticker"]].append(x)
    rows = []
    for n, t in enumerate(tickers, 1):
        ids = [c for c, _ in chunks[t]]
        texts = [tx for _, tx in chunks[t]]
        pos = {c: i for i, c in enumerate(ids)}
        g = col.get(ids=ids, include=["embeddings"])
        e = dict(zip(g["ids"], g["embeddings"], strict=True))
        qm = np.asarray([e[i] for i in ids], dtype=np.float32)
        qm /= np.linalg.norm(qm, axis=1, keepdims=True)
        nm = np.load(d.CACHE / f"{t}.npy")
        allowed = set(ids)
        for x in by_t[t]:
            if "rel" not in x:
                x["rel"] = {ids[i] for i in range(len(ids)) if passage_hit(
                    texts[i], x["gold"]["relevant_passages"], 0.5, CONTAINMENT_THRESHOLD)}
            if not x["rel"]:
                continue
            relpos = {pos[c] for c in x["rel"]}
            lex = [pos[c] for c, _ in bm25.search(x["text"], POOL, allowed)]
            dense = {"nomic": list(np.argsort(-(nm @ np.asarray(nq[x["id"]], np.float32)))[:POOL])}
            for name, p in PROMPTS.items():
                v = qwen.encode([p + x["text"]], normalize_embeddings=True)[0]
                dense[name] = list(np.argsort(-(qm @ v))[:POOL])
            variants = {"bm25_only": lex}
            for name, lst in dense.items():
                for w in BM25_W:
                    variants[f"{name}|bm25={w}"] = lst if w == 0 else rrf([lst, lex], [1 - w, w])
            for vname, lst in variants.items():
                rank = next((r for r, i in enumerate(lst, 1) if i in relpos), 0)
                rows.append({"id": x["id"], "set": x["set"], "variant": vname,
                             "hit": 0 < rank <= K, "rr": 1 / rank if rank else 0.0})
        print(f"  {n}/{len(tickers)} tickers", flush=True)
    (d.OUT / "hybrid_rows.json").write_text(json.dumps(rows))

    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for metric in ("hit", "rr"):
        t = df.pivot_table(index="variant", columns="set", values=metric, aggfunc="mean")
        t["mean_all_sets"] = t.mean(axis=1)
        print(f"\n== {'Hit@8' if metric == 'hit' else 'MRR'} ==")
        print(t.sort_values("mean_all_sets", ascending=False).round(3).to_string())
    print("\nn per set:", df[df.variant == "bm25_only"].groupby("set").size().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
