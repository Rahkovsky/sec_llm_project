#!/usr/bin/env python3
"""Q3-2 (docs/NEXT_STEPS.md): several HyDE passages in different 10-K styles.

Production retrieves with the mean cosine of the question and one company-blind HyDE passage.
FinDER answers sit in different 10-K sections, and one passage commits to one of them. This
writes three more company-blind passages, each in the style of one section family (risk
factors, MD&A, notes to the financial statements), and compares ways of combining them, all
with the production MMR (pool 60, lambda 0.85), paired against production.

    python tools/multihyde_lab.py generate --split all       # gemma4:12b, ~1 h for 260
    python tools/multihyde_lab.py evaluate --split dev

Variants:
  prod          mean cos(question, hyde)  (production)
  q+styles      mean over the question and the three styled passages
  q+hyde+styles mean over the question, the production passage and the three styled ones
  max_styles    per chunk, the best of q+hyde, q+risk, q+mdna, q+notes
  rrf_styles    reciprocal-rank fusion of the four (question + one passage) rankings
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

from rewrite_lab import (
    FINDER,
    INSTR,
    LAM,
    POOL,
    SYS,
    K,
    boot_ci,
    cache_path,
    cases_for,
    load,
    mmr,
)

OUT = Path("out/rewrite")
STYLES = {
    "risk": "the Risk Factors section (Item 1A): how the matter could adversely affect the "
            "business, results of operations or financial condition",
    "mdna": "Management's Discussion and Analysis (Item 7): what drove results, trends, "
            "liquidity and capital resources, and management's outlook",
    "notes": "the notes to the consolidated financial statements: accounting policies, "
             "estimates, contingencies, commitments and how amounts are measured",
}
P_STYLE = ("Write a short passage (3-4 sentences) in the formal style of {style} of a 10-K "
           "annual report, that would answer the question below. Use the wording companies "
           "typically use in that section. Do not invent specific numbers or names; keep it "
           "generic. Output only the passage.\n\nQuestion: {q}")


def styled_path(model: str) -> Path:
    return OUT / f"multihyde_{model.replace(':', '_')}_blind.json"


def generate(args) -> int:
    from sec_llm_project.sec_intel.llm.base import Message
    from sec_llm_project.sec_intel.llm.providers import OllamaLLM

    llm = OllamaLLM(args.model, temperature=0.0, max_tokens=300, timeout=600, num_ctx=2048)
    blind = json.loads(cache_path(args.model, True).read_text())
    path = styled_path(args.model)
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [c for c in cases_for(args.split, args.cases)
            if c["id"] in blind and c["id"] not in cache]
    print(f"{len(todo)} questions x {len(STYLES)} styles with {args.model} -> {path}", flush=True)
    for n, c in enumerate(todo, 1):
        t0, q = time.time(), blind[c["id"]]["masked_question"]  # company-blind, as production
        rec = {s: llm.generate([Message("system", SYS),
                                Message("user", P_STYLE.format(style=d, q=q))]).text.strip()
               for s, d in STYLES.items()}
        rec["seconds"] = round(time.time() - t0, 1)
        cache[c["id"]] = rec
        path.write_text(json.dumps(cache, indent=1))
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  last {rec['seconds']}s", flush=True)
    return 0


def evaluate(args) -> int:
    import chromadb
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    index = Path(args.index)
    blind = json.loads(cache_path(args.model, True).read_text())
    styled = json.loads(styled_path(args.model).read_text())
    cases = [c for c in cases_for(args.split, args.cases) if c["id"] in styled]
    tickers = sorted({c["ticker"] for c in cases})
    chunks = load(index, set(tickers))
    col = chromadb.PersistentClient(path=str(index / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float16})

    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)
    rows = []
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
            texts = [c["question"], blind[c["id"]]["hyde"], *(styled[c["id"]][s] for s in STYLES)]
            v = qwen.encode([INSTR + x for x in texts], normalize_embeddings=True)
            cos = emb @ v.T  # columns: question, hyde, risk, mdna, notes
            pairs = [(cos[:, 0] + cos[:, k]) / 2 for k in range(1, 5)]
            scores = {"prod": pairs[0],
                      "q+styles": cos[:, [0, 2, 3, 4]].mean(axis=1),
                      "q+hyde+styles": cos.mean(axis=1),
                      "max_styles": np.max(pairs, axis=0)}
            rank = np.zeros(len(meta))
            for p in pairs:
                r = np.empty(len(meta))
                r[np.argsort(-p)] = np.arange(1, len(meta) + 1)
                rank += 1 / (60 + r)
            scores["rrf_styles"] = rank
            for name, s in scores.items():
                order = list(np.argsort(-s))
                # MMR needs a relevance on the cosine scale; RRF ranks reuse the q+hyde score
                rel_s = pairs[0] if name == "rrf_styles" else s
                lst = mmr(order[:POOL], rel_s, emb, LAM, K)
                r = next((k for k, i in enumerate(lst, 1) if i in rel), 0)
                rows.append({"id": c["id"], "variant": name, "hit": 0 < r <= K,
                             "rr": 1 / r if r else 0.0})

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
    txt = (f"{index} split={args.split}: scorable {len(piv)}; MMR pool {POOL}, lambda {LAM}; "
           f"Delta vs prod\n\n" + tab.to_string())
    (OUT / f"multihyde_{index.name}_{args.split}.txt").write_text(txt)
    print(txt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["generate", "evaluate"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--index", default="data/sec_index_v4")
    ap.add_argument("--cases", default=FINDER)
    args = ap.parse_args()
    return generate(args) if args.phase == "generate" else evaluate(args)


if __name__ == "__main__":
    sys.exit(main())
