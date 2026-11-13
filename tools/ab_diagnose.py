#!/usr/bin/env python3
"""Diagnose WHY Qwen3-Embedding-0.6B retrieves worse than Nomic (hypothesis tests).

Runs on the tickers already embedded by ``tools/ab_embedders.py --nomic-only``
(cached Nomic doc vectors; Qwen doc vectors come from the production Chroma).

Hypotheses
  H1 query-style mismatch : Qwen hurt by FinDER's terse/filler statements, not by
                            the corpus. Test: 2x2 {model} x {query style} on the
                            SAME targets (synthetic known-item queries, natural vs
                            FinDER-style), plus FinDER queries rewritten as questions.
  H2 pooling x chunk length: last-token pooling dilutes long multi-topic chunks.
                            Test: hit rate binned by target-chunk length.
  H3 discrimination       : Qwen's within-ticker scores are bunched, so the
                            right chunk loses to near-duplicates. Test: score
                            spread, anisotropy, standardized gold margin.

Phases (run in order):
  python tools/make_synth_qa.py --submit / --collect    # Claude writes the queries
  <nomic_env>/python tools/ab_diagnose.py --encode-nomic  # queries -> Nomic vectors
  python tools/ab_diagnose.py                           # analysis -> out/ab_diag/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

INDEX = Path("data/sec_index_qwen06")
CACHE = Path("out/ab_embedders/nomic")
OUT = Path("out/ab_diag")
NOMIC = ("nomic-ai/nomic-embed-text-v1.5", "e9b6763023c676ca8431644204f50c2b100d9aab")
QWEN_PROMPTS = {
    "qwen_instr0": "Instruct: Given a question about SEC filings, retrieve the passages that "
                   "answer it\nQuery: ",
    "qwen_instr1": "Instruct: Given a terse search phrase from a financial analyst about a "
                   "company, retrieve the passage of the company's 10-K annual report that "
                   "answers it\nQuery: ",
    "qwen_web": "Instruct: Given a web search query, retrieve relevant passages that answer "
                "the query\nQuery:",  # the model card's default prompt
    "qwen_none": "",
}
K = 8


def cached_tickers() -> list[str]:
    return sorted(p.stem for p in CACHE.glob("*.npy"))


def load_chunks(tickers: set[str]) -> dict[str, list[tuple[str, str]]]:
    chunks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(INDEX / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            t = d["metadata"].get("ticker")
            if t in tickers:
                chunks[t].append((d["chunk_id"], d["text"]))
    return chunks


def query_sets() -> list[dict]:
    """Every query: id, set, ticker, text, relevant ids (filled later for finder)."""
    q = json.loads((OUT / "queries.json").read_text())
    dev = set(json.loads(Path("eval/finder_split.json").read_text())["dev"])
    tick = set(cached_tickers())
    finder = [c for c in json.loads(Path("eval/finder_semantic.json").read_text())
              if c["ticker"] in tick and c["id"] in dev]
    out = []
    for c in finder:
        out.append({"id": c["id"], "set": "finder", "ticker": c["ticker"], "text": c["question"],
                    "gold": c})
        if c["id"] in q["finder_q"]:
            out.append({"id": c["id"] + "#q", "set": "finder_as_question", "ticker": c["ticker"],
                        "text": q["finder_q"][c["id"]], "gold": c})
    for i, s in enumerate(q["synth"]):
        for style in ("question", "terse"):
            out.append({"id": f"synth{i}_{style}", "set": f"synth_{style}", "ticker": s["ticker"],
                        "text": s[style], "rel": set(s.get("rel_ids") or [s["chunk_id"]])})
    return out


def encode_nomic() -> None:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(NOMIC[0], revision=NOMIC[1], trust_remote_code=True)
    qs = query_sets()
    v = m.encode([f"search_query: {x['text']}" for x in qs], batch_size=32,
                 normalize_embeddings=True)
    (OUT / "nomic_q.json").write_text(json.dumps({x["id"]: vv.tolist() for x, vv in zip(qs, v)}))
    print(f"nomic: {len(qs)} queries encoded")


def analyze() -> None:  # noqa: PLR0912, PLR0915 - linear analysis
    import chromadb
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    qs = query_sets()
    nq = json.loads((OUT / "nomic_q.json").read_text())
    tickers = sorted({x["ticker"] for x in qs})
    chunks = load_chunks(set(tickers))
    length = {cid: len(tx) for lst in chunks.values() for cid, tx in lst}
    col = chromadb.PersistentClient(path=str(INDEX / "chroma_db")).get_collection("sec_filings")
    import torch
    # fp32 queries: bf16 query noise (~0.001) is ~60% of the rank-8/9 score gap.
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                               model_kwargs={"dtype": torch.float32})

    def norm(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    rows = []
    aniso = defaultdict(list)
    by_t = defaultdict(list)
    for x in qs:
        by_t[x["ticker"]].append(x)
    for n, t in enumerate(tickers, 1):
        ids = [cid for cid, _ in chunks[t]]
        texts = [tx for _, tx in chunks[t]]
        got = col.get(ids=ids, include=["embeddings"])
        qv = dict(zip(got["ids"], got["embeddings"], strict=True))
        mats = {"qwen": norm(np.asarray([qv[i] for i in ids], dtype=np.float32)),
                "nomic": np.load(CACHE / f"{t}.npy")}
        rs = np.random.default_rng(0).choice(len(ids), min(300, len(ids)), replace=False)
        for name, m in mats.items():  # anisotropy: mean pairwise cosine within ticker
            s = m[rs] @ m[rs].T
            aniso[name].append(float(s[np.triu_indices(len(rs), 1)].mean()))
        for x in by_t[t]:
            if "rel" not in x:  # finder: every chunk satisfying the cover rule
                x["rel"] = {ids[i] for i in range(len(ids)) if passage_hit(
                    texts[i], x["gold"]["relevant_passages"], 0.5, CONTAINMENT_THRESHOLD)}
            if not x["rel"]:
                continue
            relpos = [i for i, c in enumerate(ids) if c in x["rel"]]
            vecs = {"nomic": ("nomic", np.asarray(nq[x["id"]], dtype=np.float32))}
            for name, p in QWEN_PROMPTS.items():
                vecs[name] = ("qwen", qwen.encode([p + x["text"]], normalize_embeddings=True)[0])
            for name, (space, v) in vecs.items():
                s = mats[space] @ v
                order = np.argsort(-s)
                rank = int(np.where(np.isin(order, relpos))[0][0]) + 1
                best = max(relpos, key=lambda i: s[i])
                rows.append({"id": x["id"], "set": x["set"], "ticker": t, "model": name,
                             "rank": rank, "hit": rank <= K, "rr": 1 / rank,
                             "z_margin": float((s[best] - s[order[K - 1]]) / s.std()),
                             "spread": float(s.std()), "top1_minus_top8": float(
                                 s[order[0]] - s[order[K - 1]]),
                             "rel_len": float(np.mean([length[ids[i]] for i in relpos])),
                             "n_rel": len(relpos)})
        print(f"  {n}/{len(tickers)} tickers", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows))

    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n== H1: Hit@8 / MRR by query set x model ==")
    g = df.groupby(["set", "model"]).agg(n=("hit", "size"), hit=("hit", "mean"),
                                          mrr=("rr", "mean"), med_rank=("rank", "median"))
    print(g.unstack("model")[["hit"]].round(3).to_string())
    print(g.unstack("model")[["mrr"]].round(3).to_string())
    print(g.unstack("model")[["med_rank"]].to_string())
    print("\n== H2: Hit@8 by mean relevant-chunk length (chars) ==")
    df["len_bin"] = pd.cut(df["rel_len"], [0, 1500, 2500, 3200, 10**6],
                           labels=["<1.5k", "1.5-2.5k", "2.5-3.2k", ">3.2k"])
    for st in ("synth_question", "finder"):
        sub = df[(df["set"] == st) & df["model"].isin(["nomic", "qwen_instr0", "qwen_instr1"])]
        print(f"-- {st}")
        print(sub.pivot_table(index="len_bin", columns="model", values="hit", aggfunc="mean",
                              observed=False).round(3).to_string())
        print(sub[sub.model == "nomic"].groupby("len_bin", observed=False).size().to_string())
    print("\n== H3: score geometry (median over queries) ==")
    print(df.groupby(["model"])[["spread", "top1_minus_top8", "z_margin"]].median()
          .round(4).to_string())
    print("anisotropy (mean within-ticker doc cosine):",
          {k: round(float(np.mean(v)), 3) for k, v in aniso.items()})
    print("\n== paired: per query, which model ranks the target better (synth_question) ==")
    p = df[df.set == "synth_question"].pivot_table(index="id", columns="model", values="rank")
    for m in ("qwen_instr0", "qwen_instr1"):
        print(f"{m}: better {int((p[m] < p['nomic']).sum())}, worse "
              f"{int((p[m] > p['nomic']).sum())}, tie {int((p[m] == p['nomic']).sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encode-nomic", action="store_true")
    args = ap.parse_args()
    if args.encode_nomic:
        encode_nomic()
    else:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
