#!/usr/bin/env python3
"""Head-to-head embedder A/B on identical chunks (Nomic vs Qwen3-Embedding).

The old Nomic index no longer exists, but retrieval is always filtered to one
ticker, so a fair comparison only needs each eval ticker's chunks: they are
embedded with Nomic (cached under out/ab_embedders/nomic/<TICKER>.npy) and
searched EXACTLY (brute-force cosine), alongside Qwen using the vectors already
stored in Chroma — both exact and via Chroma's approximate HNSW query, which
isolates any ANN loss under the ticker filter. Hybrid rows fuse with the
production BM25 index (bm25.json) by weighted RRF, reproducing the old
local.yaml retrieval (BM25 0.6 / dense 0.4) on today's corpus.

    python tools/ab_embedders.py --split dev
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

from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit
from sec_llm_project.sec_intel.retrieval.bm25 import BM25Index
from sec_llm_project.sec_intel.retrieval.fusion import reciprocal_rank_fusion

INDEX = Path("data/sec_index_qwen06")
NOMIC = ("nomic-ai/nomic-embed-text-v1.5", "e9b6763023c676ca8431644204f50c2b100d9aab")
QWEN_INSTR = {
    "instr0": "Instruct: Given a question about SEC filings, retrieve the passages that "
              "answer it\nQuery: ",
    "instr1": "Instruct: Given a terse search phrase from a financial analyst about a "
              "company, retrieve the passage of the company's 10-K annual report that "
              "answers it\nQuery: ",
}
K, POOL = 8, 60


def main() -> int:  # noqa: PLR0912 - linear experiment script
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--max-tickers", type=int, default=0, help="pilot on the first N tickers")
    ap.add_argument("--out", default="out/ab_embedders")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Nomic encode batch; its remote code materializes full attention, so "
                         "32 x ~900-token chunks pins ~6 GB of MPS (wired) memory on 16 GB")
    ap.add_argument("--nomic-only", action="store_true",
                    help="only embed chunks + queries with Nomic into the cache. Run this in an "
                         "env with the uv.lock transformers (Nomic's pinned remote code needs it)")
    args = ap.parse_args()

    split = json.loads(Path("eval/finder_split.json").read_text())
    keep = set(split["dev"] + split["test"]) if args.split == "all" else set(split[args.split])
    cases = [c for c in json.loads(Path(args.cases).read_text()) if c["id"] in keep]
    tickers = sorted({c["ticker"] for c in cases})
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
        cases = [c for c in cases if c["ticker"] in set(tickers)]
    print(f"{args.split}: {len(cases)} questions over {len(tickers)} tickers", flush=True)

    # ---- chunk texts for those tickers (same text both models embedded)
    chunks: dict[str, list[tuple[str, str]]] = defaultdict(list)
    want = set(tickers)
    with open(INDEX / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            t = d["metadata"].get("ticker")
            if t in want:
                chunks[t].append((d["chunk_id"], d["text"]))
    text = {cid: tx for lst in chunks.values() for cid, tx in lst}
    print(f"chunks: {len(text)}", flush=True)

    from sentence_transformers import SentenceTransformer
    cache = Path(args.out) / "nomic"
    cache.mkdir(parents=True, exist_ok=True)
    qcache_path = cache / "queries.json"
    if args.nomic_only:
        nomic = SentenceTransformer(NOMIC[0], revision=NOMIC[1], trust_remote_code=True)
        t0 = time.perf_counter()
        for n, t in enumerate(tickers, 1):
            f = cache / f"{t}.npy"
            if not (f.exists() and len(np.load(f)) == len(chunks[t])):
                np.save(f, nomic.encode([f"search_document: {tx}" for _, tx in chunks[t]],
                                        batch_size=args.batch_size, normalize_embeddings=True
                                        ).astype(np.float32))
            if n % 5 == 0 or n == len(tickers):
                print(f"  nomic {n}/{len(tickers)} tickers ({time.perf_counter() - t0:.0f}s)",
                      flush=True)
        qc = json.loads(qcache_path.read_text()) if qcache_path.exists() else {}
        for c in cases:
            if c["id"] not in qc:
                qc[c["id"]] = nomic.encode([f"search_query: {c['question']}"],
                                           normalize_embeddings=True)[0].tolist()
        qcache_path.write_text(json.dumps(qc))
        print("nomic cache complete")
        return 0
    qcache = json.loads(qcache_path.read_text())

    import chromadb
    col = chromadb.PersistentClient(path=str(INDEX / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    print("loading BM25...", flush=True)
    bm25 = BM25Index.from_saved(INDEX / "bm25.json")

    def norm(m: np.ndarray) -> np.ndarray:
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    by_ticker = defaultdict(list)
    for c in cases:
        by_ticker[c["ticker"]].append(c)
    ranked: dict[str, dict[str, list[str]]] = defaultdict(dict)  # variant -> qid -> ids
    t0 = time.perf_counter()
    for n, t in enumerate(tickers, 1):
        ids = [cid for cid, _ in chunks[t]]
        # Qwen: stored vectors (exactly what production searches)
        got = col.get(ids=ids, include=["embeddings"])
        qv = dict(zip(got["ids"], got["embeddings"], strict=True))
        qmat = norm(np.asarray([qv[i] for i in ids], dtype=np.float32))
        # Nomic: embed once per ticker, cached
        nmat = np.load(cache / f"{t}.npy")
        assert len(nmat) == len(ids), f"stale nomic cache for {t}; rerun --nomic-only"
        allowed = set(ids)
        for c in by_ticker[t]:
            q = c["question"]
            dense = {}
            for name, ins in QWEN_INSTR.items():
                v = qwen.encode([ins + q], normalize_embeddings=True)[0]
                order = np.argsort(-(qmat @ v))[:POOL]
                dense[f"qwen_{name}"] = [ids[i] for i in order]
                if name == "instr0":
                    res = col.query(query_embeddings=[v.tolist()], n_results=K,
                                    where={"ticker": t})
                    ranked["qwen_instr0_chroma (production)"][c["id"]] = res["ids"][0]
            v = np.asarray(qcache[c["id"]], dtype=np.float32)
            dense["nomic"] = [ids[i] for i in np.argsort(-(nmat @ v))[:POOL]]
            lex = [cid for cid, _ in bm25.search(q, POOL, allowed)]
            for name, lst in dense.items():
                ranked[f"{name}_exact"][c["id"]] = lst[:K]
            ranked["bm25_only"][c["id"]] = lex[:K]
            for name in ("nomic", "qwen_instr0", "qwen_instr1"):
                fused = reciprocal_rank_fusion([dense[name], lex], k=60, weights=[0.4, 0.6])
                top = sorted(fused.items(), key=lambda x: x[1], reverse=True)
                ranked[f"{name}+bm25 0.6/0.4"][c["id"]] = [cid for cid, _ in top[:K]]
        if n % 10 == 0 or n == len(tickers):
            print(f"  {n}/{len(tickers)} tickers ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- score
    rel: dict[tuple[str, str, str], bool] = {}
    cats = sorted({c["category"] for c in cases})
    print(f"\n{'variant':34s} {'Hit@8 cover':>11} {'MRR':>6} {'Hit@8 strict':>12}   "
          + "  ".join(f"{c[:8]:>8}" for c in cats))
    rows = []
    for name, per in ranked.items():
        m = {"cover": [0, 0.0], "strict": [0, 0.0]}
        catn = defaultdict(int)
        for c in cases:
            for rule, cont in (("cover", CONTAINMENT_THRESHOLD), ("strict", None)):
                first = 0
                for i, cid in enumerate(per[c["id"]], 1):
                    key = (c["id"], cid, rule)
                    if key not in rel:
                        rel[key] = passage_hit(text.get(cid) or "", c["relevant_passages"],
                                               0.5, cont)
                    if rel[key]:
                        first = i
                        break
                m[rule][0] += first > 0
                m[rule][1] += 1 / first if first else 0.0
                if rule == "cover" and first:
                    catn[c["category"]] += 1
        n = len(cases)
        ncat = {k: sum(c["category"] == k for c in cases) for k in cats}
        print(f"{name:34s} {m['cover'][0] / n:>11.3f} {m['cover'][1] / n:>6.3f} "
              f"{m['strict'][0] / n:>12.3f}   "
              + "  ".join(f"{catn[k] / ncat[k]:>8.2f}" for k in cats))
        rows.append({"variant": name, "hit_cover": m["cover"][0] / n,
                     "mrr_cover": m["cover"][1] / n, "hit_strict": m["strict"][0] / n,
                     **{f"cover_{k}": catn[k] / ncat[k] for k in cats}})
        (Path(args.out) / f"perq_{args.split}_{name.split()[0]}.json").write_text(
            json.dumps({cid: per[cid] for cid in per}))
    (Path(args.out) / f"summary_{args.split}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
