#!/usr/bin/env python3
"""Period filters: does restricting (or boosting) the filings of the named year help?

Questions that name a year ("in fiscal 2023", "in 2024") are answered from the
ticker's filings of every year; the evidence can come from the wrong period
(docs/EXPERIMENTS.md §9, wrong-period answers). This lab takes the production
retriever (mean cos to the question and its company-blind HyDE passage, inside
the ticker filter, MMR lambda 0.85 over the top 60) and compares, paired, on the
year-named questions:

  prod          ticker filter only (production)
  hard_y        only filings with fiscal_year == Y (fallback: no year filter)
  hard_y1       fiscal_year in {Y, Y+1} (a 10-K for Y+1 restates Y as comparative)
  soft_<d>      +d to the cosine of chunks in {Y, Y+1} before pooling/MMR

Metrics: Hit@8 / MRR@8 of the gold passage, and the share of the top 8 from the
named period ({Y, Y+1}).

    python tools/period_lab.py --cases eval/synth_claude.json
    python tools/period_lab.py   # FinDER (few year-named questions)
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

from rewrite_lab import FINDER, INSTR, K, LAM, POOL, boot_ci, cache_path, cases_for  # noqa: E402
from diversity_lab import mmr  # noqa: E402

from sec_llm_project.sec_intel.retrieval.period import named_years  # noqa: E402

OUT = Path("out/period")


def load(index: Path, tickers: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with open(index / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            m = d["metadata"]
            if m.get("ticker") in tickers:
                out[m["ticker"]].append({"id": d["chunk_id"], "text": d["text"],
                                         "fy": str(m.get("fiscal_year", "")),
                                         "form": m.get("filing_type", "")})
    return out


def main() -> int:
    import chromadb
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/sec_index_v3")
    ap.add_argument("--cases", default=FINDER)
    ap.add_argument("--model", default="gemma4:12b")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = Path(args.index)
    blind = json.loads(cache_path(args.model, True).read_text())
    cases = [c for c in cases_for("all", args.cases)
             if c["id"] in blind and c.get("ticker") and named_years(c["question"])]
    tickers = sorted({c["ticker"] for c in cases})
    print(f"{index} {Path(args.cases).stem}: {len(cases)} year-named questions", flush=True)
    chunks = load(index, set(tickers))
    col = chromadb.PersistentClient(path=str(index / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float16})

    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)
    rows, src_fy = [], []
    for t in tickers:
        meta = chunks[t]
        if not meta:
            continue
        ids = [m["id"] for m in meta]
        got = col.get(ids=ids, include=["embeddings"])
        e = dict(zip(got["ids"], got["embeddings"], strict=True))
        emb = np.asarray([e[i] for i in ids], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        fy = np.asarray([m["fy"] for m in meta])
        for c in by_t[t]:
            rel = {i for i, m in enumerate(meta)
                   if passage_hit(m["text"], c["relevant_passages"], 0.5, CONTAINMENT_THRESHOLD)}
            if not rel:
                continue
            years = named_years(c["question"])
            exact = {str(y) for y in years}
            wide = exact | {str(y + 1) for y in years}
            in_exact = np.isin(fy, list(exact))
            in_wide = np.isin(fy, list(wide))
            src = c.get("source", "").rsplit(" from ", 1)[-1]
            src_i = next((i for i, m in enumerate(meta) if m["id"] == src), None)
            if src_i is not None:
                src_fy.append(int(meta[src_i]["fy"]) - max(years) if meta[src_i]["fy"].isdigit()
                              else None)
            v = qwen.encode([INSTR + c["question"], INSTR + blind[c["id"]]["hyde"]],
                            normalize_embeddings=True)
            s = (emb @ v.T).mean(axis=1)

            def top(score: np.ndarray, mask: np.ndarray | None) -> list[int]:
                sc = score if mask is None or not mask.any() else np.where(mask, score, -9.0)
                order = np.argsort(-sc)
                pool = [int(i) for i in order[:POOL] if sc[i] > -9.0]
                return mmr(pool, sc, emb, LAM, K)

            variants = {"prod": top(s, None), "hard_y": top(s, in_exact),
                        "hard_y1": top(s, in_wide)}
            for d in (0.02, 0.05, 0.1):
                variants[f"soft_{d}"] = top(s + d * in_wide, None)
            for name, lst in variants.items():
                r = next((k for k, i in enumerate(lst, 1) if i in rel), 0)
                rows.append({"id": c["id"], "variant": name, "hit": 0 < r <= K,
                             "rr": 1 / r if r else 0.0,
                             "in_period": float(np.mean([in_wide[i] for i in lst]))
                             if lst else 0.0,
                             "gold_in_period": any(in_wide[i] for i in rel)})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="id", columns="variant", values=["hit", "rr", "in_period"])
    base = piv["hit"]["prod"].astype(float)
    res = []
    for v in df.variant.unique():
        hit = piv["hit"][v].astype(float)
        dh = (hit - base).to_numpy()
        lo, hi = boot_ci(dh)
        res.append({"variant": v, "Hit@8": hit.mean(), "MRR@8": piv["rr"][v].mean(),
                    "in-period@8": piv["in_period"][v].mean(), "dHit": dh.mean(),
                    "CI95": f"[{lo:+.3f},{hi:+.3f}]",
                    "W/L": f"{int((dh > 0).sum())}/{int((dh < 0).sum())}"})
    pd.set_option("display.width", 200)
    tab = pd.DataFrame(res).set_index("variant").sort_values("Hit@8", ascending=False).round(3)
    gin = df[df.variant == "prod"].gold_in_period.mean()
    off = pd.Series([x for x in src_fy if x is not None]).value_counts().sort_index()
    stem = f"{index.name}_{Path(args.cases).stem}"
    txt = (f"{index} {Path(args.cases).stem}: scorable year-named {len(piv)}; gold passage in a "
           f"filing of {{Y, Y+1}}: {gin:.2f}\nsource filing fiscal_year - named year: "
           f"{dict(off)}\n\n" + tab.to_string())
    (OUT / f"{stem}.txt").write_text(txt)
    df.to_json(OUT / f"{stem}_rows.json", orient="records")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
