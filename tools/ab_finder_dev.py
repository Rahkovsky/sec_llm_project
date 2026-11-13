#!/usr/bin/env python3
"""All FinDER dev questions: Qwen prompts x BM25 hybrid (+ Nomic where cached).

tools/ab_hybrid.py only covers the 40 tickers with cached Nomic vectors (46
scorable FinDER dev questions). This runs every FinDER dev question (128, 115
tickers) with production-equivalent retrieval inside the ticker filter:

  * dense Qwen: stored Chroma doc vectors, fp32 query encoder, 4 prompts
  * BM25: per-ticker index re-weighted with the GLOBAL corpus statistics (idf,
    avgdl, N streamed once from chunks.jsonl and cached), so scores equal the
    production bm25.json without loading its ~5 GB
  * hybrid: weighted RRF (k=60) of the top-60 lists, as retrieval/hybrid.py
  * Nomic: wherever tools/ab_embedders.py --nomic-only has cached the ticker

Relevance = chunks passing passage_hit against FinDER's relevant_passages
(the same rule as the other tools). Reports Hit@8 / MRR on all scorable
questions and on the Nomic-covered subset, with paired bootstrap 95% CIs of
each variant minus qwen_instr1 dense.

    python tools/ab_finder_dev.py      # -> out/ab_diag/finder_dev.txt, finder_dev_rows.json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import ab_diagnose as d  # noqa: E402
from ab_hybrid import rrf  # noqa: E402

PROMPTS = {k: d.QWEN_PROMPTS[k] for k in ("qwen_none", "qwen_instr0", "qwen_instr1", "qwen_web")}
BM25_W = [0.0, 0.3, 0.5, 0.7]
POOL, K = 60, 8
GLOBAL = d.OUT / "bm25_global.json"
NOMIC_Q = Path("out/ab_embedders/nomic/queries.json")
BASE = "qwen_instr1|bm25=0.0"


def global_stats() -> dict:
    """N, avgdl and idf over the whole corpus, exactly as BM25Index.__init__."""
    if GLOBAL.exists():
        return json.loads(GLOBAL.read_text())
    from sec_llm_project.sec_intel.retrieval.bm25 import tokenize
    df: Counter[str] = Counter()
    n = total = 0
    with open(d.INDEX / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            toks = tokenize(json.loads(line)["text"])
            df.update(set(toks))
            n += 1
            total += len(toks)
            if n % 100_000 == 0:
                print(f"  bm25 stats {n} chunks", flush=True)
    st = {"n": n, "avgdl": total / n,
          "idf": {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}}
    GLOBAL.write_text(json.dumps(st))
    return st


def ticker_bm25(chunks: list[tuple[str, str]], st: dict):
    from sec_llm_project.sec_intel.core.types import Chunk
    from sec_llm_project.sec_intel.retrieval.bm25 import BM25Index
    idx = BM25Index([Chunk(chunk_id=c, text=t, metadata={}) for c, t in chunks])
    # global idf/avgdl give production scores; _n stays local (search iterates range(_n))
    idx._avgdl, idx._idf = st["avgdl"], st["idf"]
    return idx


def boot_ci(diff: np.ndarray, reps: int = 5000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), (reps, len(diff)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(df, title: str) -> str:
    import pandas as pd
    out = [f"\n===== {title} ====="]
    for s in sorted(df.set.unique()):
        sub = df[df.set == s]
        piv = sub.pivot_table(index="id", columns="variant", values=["hit", "rr"])
        n = len(piv)
        rows = []
        for v in sorted(sub.variant.unique()):
            hit, rr = piv["hit"][v].astype(float), piv["rr"][v]
            row = {"variant": v, "Hit@8": hit.mean(), "MRR": rr.mean()}
            if BASE in piv["hit"] and v != BASE:
                dh = (hit - piv["hit"][BASE].astype(float)).to_numpy()
                lo, hi = boot_ci(dh)
                row.update({"dHit_vs_instr1": dh.mean(), "CI95": f"[{lo:+.3f},{hi:+.3f}]",
                            "W/L": f"{int((dh > 0).sum())}/{int((dh < 0).sum())}"})
            rows.append(row)
        t = pd.DataFrame(rows).set_index("variant").sort_values("Hit@8", ascending=False)
        out.append(f"\n-- set={s}  n={n} --\n" + t.round(3).to_string())
    return "\n".join(out)


def main() -> int:
    import chromadb
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    dev = set(json.loads(Path("eval/finder_split.json").read_text())["dev"])
    cases = [c for c in json.loads(Path("eval/finder_semantic.json").read_text()) if c["id"] in dev]
    finder_q = json.loads((d.OUT / "queries.json").read_text())["finder_q"]
    nomic_q = json.loads(NOMIC_Q.read_text()) if NOMIC_Q.exists() else {}
    nomic_q_rw = json.loads((d.OUT / "nomic_q.json").read_text())  # rewrites: "<id>#q"
    tickers = sorted({c["ticker"] for c in cases})
    print(f"FinDER dev: {len(cases)} questions, {len(tickers)} tickers", flush=True)

    st = global_stats()
    chunks = d.load_chunks(set(tickers))
    col = chromadb.PersistentClient(path=str(d.INDEX / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float32})

    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)
    rows, unscorable = [], []
    for n, t in enumerate(tickers, 1):
        ids = [c for c, _ in chunks[t]]
        texts = [tx for _, tx in chunks[t]]
        g = col.get(ids=ids, include=["embeddings"])
        e = dict(zip(g["ids"], g["embeddings"], strict=True))
        qm = np.asarray([e[i] for i in ids], dtype=np.float32)
        qm /= np.linalg.norm(qm, axis=1, keepdims=True)
        npy = d.CACHE / f"{t}.npy"
        nm = np.load(npy) if npy.exists() else None
        if nm is not None and len(nm) != len(ids):
            nm = None
        bm = ticker_bm25(chunks[t], st)
        pos = {c: i for i, c in enumerate(ids)}
        for c in by_t[t]:
            rel = {i for i in range(len(ids))
                   if passage_hit(texts[i], c["relevant_passages"], 0.5, CONTAINMENT_THRESHOLD)}
            if not rel:
                unscorable.append(f"{c['id']} {t}: {c['question'][:80]}")
                continue
            queries = [("finder", c["id"], c["question"], nomic_q.get(c["id"]))]
            if c["id"] in finder_q:
                queries.append(("finder_as_question", c["id"], finder_q[c["id"]],
                                nomic_q_rw.get(c["id"] + "#q")))
            for set_name, qid, text, nvec in queries:
                lex = [pos[cid] for cid, _ in bm.search(text, POOL)]
                dense = {}
                for name, p in PROMPTS.items():
                    v = qwen.encode([p + text], normalize_embeddings=True)[0]
                    dense[name] = list(np.argsort(-(qm @ v))[:POOL])
                if nm is not None and nvec is not None:
                    dense["nomic"] = list(np.argsort(-(nm @ np.asarray(nvec, np.float32)))[:POOL])
                variants = {"bm25_only": lex}
                for name, lst in dense.items():
                    for w in BM25_W:
                        variants[f"{name}|bm25={w}"] = lst if w == 0 else rrf([lst, lex], [1 - w, w])
                for vname, lst in variants.items():
                    rank = next((r for r, i in enumerate(lst, 1) if i in rel), 0)
                    rows.append({"id": qid, "set": set_name, "ticker": t, "variant": vname,
                                 "nomic_covered": nm is not None and nvec is not None,
                                 "hit": 0 < rank <= K, "rr": 1 / rank if rank else 0.0})
        if n % 10 == 0 or n == len(tickers):
            print(f"  {n}/{len(tickers)} tickers", flush=True)
    (d.OUT / "finder_dev_rows.json").write_text(json.dumps(rows))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    cov = df[df.nomic_covered].id.unique()
    txt = [f"FinDER dev: {len(cases)} questions; scorable {df[df.set == 'finder'].id.nunique()}; "
           f"Nomic-covered {df[(df.set == 'finder') & df.nomic_covered].id.nunique()}",
           "Unscorable (no chunk contains the gold evidence):", *[f"  {u}" for u in unscorable],
           report(df[~df.variant.str.startswith("nomic")], "ALL scorable questions (Qwen/BM25)")]
    if len(cov):
        txt.append(report(df[df.id.isin(cov)], "Nomic-covered subset (all variants)"))
    out = "\n".join(txt)
    (d.OUT / "finder_dev.txt").write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
