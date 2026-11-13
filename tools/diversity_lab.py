#!/usr/bin/env python3
"""Form-agnostic diversification of the top-k: does it lift FinDER Hit@8?

Production retrieval = Qwen3-Embedding-0.6B dense, "instr1" query prompt,
inside the question's ticker filter, top 8. Every variant below re-orders the
same dense candidate pool using only chunk text / embeddings (no section
labels, no form type), so it works the same on 10-K, 10-Q and 8-K:

  base            dense top 8 (production)
  dedup[best|new] collapse near-duplicates (token Jaccard >= JACC) in the pool;
                  the group takes its best score; representative = the best-
                  scoring member or the most recently filed one
  mmr@L           dedup(best) then MMR: L*sim(q,d) - (1-L)*max sim(d, chosen)
  cap@N           dedup(best) then at most N chunks per filing (accession)
  clus@P          dedup(best) then k-means(8) over the top-P pool; best chunk of
                  each cluster, clusters ordered by their best score

Relevance = passage_hit vs FinDER relevant_passages (as the other tools).
Reports Hit@8 / MRR@8 with paired bootstrap CIs vs base, plus diversity of the
top 8 (distinct filings, distinct section labels -- labels only DESCRIBE the
result, they are not used to rank).

    python tools/diversity_lab.py [--index data/sec_index_qwen06] [--split dev]
        -> out/diversity/<index>_<split>.txt, _rows.json
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

from ab_diagnose import QWEN_PROMPTS  # noqa: E402
from ab_finder_dev import boot_ci  # noqa: E402

K = 8
POOL = 200
JACC = 0.8
PROMPT = QWEN_PROMPTS["qwen_instr1"]


def load(index: Path, tickers: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with open(index / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            m = d["metadata"]
            if m.get("ticker") in tickers:
                out[m["ticker"]].append({"id": d["chunk_id"], "text": d["text"],
                                         "acc": m.get("accession", ""),
                                         "date": m.get("filing_date") or "",
                                         "form": m.get("filing_type", "")})
    return out


def dup_groups(pool: list[int], toks: list[set], jacc: float) -> list[list[int]]:
    """Greedy grouping in score order: a chunk joins the first group whose head it matches."""
    groups: list[list[int]] = []
    for i in pool:
        for g in groups:
            a, b = toks[i], toks[g[0]]
            if a and b and len(a & b) / len(a | b) >= jacc:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def dedup(pool, toks, meta, keep: str) -> list[int]:
    reps = []
    for g in dup_groups(pool, toks, JACC):  # groups come out in best-score order
        if keep == "new":
            # newest filing; FinDER gold pseudo-chunks (date UNKNOWN) count as oldest
            reps.append(max(g, key=lambda i: (meta[i]["date"][:4].isdigit(), meta[i]["date"])))
        else:
            reps.append(g[0])
    return reps


def mmr(cands: list[int], s: np.ndarray, emb: np.ndarray, lam: float, k: int) -> list[int]:
    chosen: list[int] = []
    rest = list(cands)
    while rest and len(chosen) < k:
        if chosen:
            red = (emb[rest] @ emb[chosen].T).max(axis=1)
            val = lam * s[rest] - (1 - lam) * red
        else:
            val = s[rest]
        j = rest[int(np.argmax(val))]
        chosen.append(j)
        rest.remove(j)
    return chosen


def cap(cands: list[int], meta, n: int, k: int) -> list[int]:
    seen: dict[str, int] = defaultdict(int)
    out = []
    for i in cands:
        if seen[meta[i]["acc"]] < n:
            seen[meta[i]["acc"]] += 1
            out.append(i)
            if len(out) == k:
                break
    return out


def cluster_pick(cands: list[int], s: np.ndarray, emb: np.ndarray, k: int) -> list[int]:
    from sklearn.cluster import KMeans
    if len(cands) <= k:
        return cands
    lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(emb[cands])
    best: dict[int, int] = {}
    for i, c in zip(cands, lab, strict=True):  # cands are in score order: first = best
        best.setdefault(int(c), i)
    return sorted(best.values(), key=lambda i: -s[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/sec_index_qwen06")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--report-only", action="store_true", help="re-report from saved rows")
    args = ap.parse_args()
    if args.report_only:
        return report(Path(args.index), args.split, None, [])
    import chromadb
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

    index = Path(args.index)
    split = json.loads(Path("eval/finder_split.json").read_text())
    keep = set(split["dev"]) | set(split["test"]) if args.split == "all" else set(split[args.split])
    cases = [c for c in json.loads(Path("eval/finder_semantic.json").read_text()) if c["id"] in keep]
    tickers = sorted({c["ticker"] for c in cases})
    print(f"{index} {args.split}: {len(cases)} questions, {len(tickers)} tickers", flush=True)
    chunks = load(index, set(tickers))
    col = chromadb.PersistentClient(path=str(index / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float32})
    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)

    rows, unscorable = [], []
    for n, t in enumerate(tickers, 1):
        meta = chunks[t]
        ids = [m["id"] for m in meta]
        got = col.get(ids=ids, include=["embeddings", "metadatas"])
        e = dict(zip(got["ids"], got["embeddings"], strict=True))
        item = {i: (md or {}).get("item_number", "?") for i, md in zip(got["ids"], got["metadatas"],
                                                                       strict=True)}
        emb = np.asarray([e[i] for i in ids], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        toks = [set(m["text"].lower().split()) for m in meta]
        for c in by_t[t]:
            rel = {i for i, m in enumerate(meta)
                   if passage_hit(m["text"], c["relevant_passages"], 0.5, CONTAINMENT_THRESHOLD)}
            if not rel:
                unscorable.append(c["id"])
                continue
            q = qwen.encode([PROMPT + c["question"]], normalize_embeddings=True)[0]
            s = emb @ q
            order = list(np.argsort(-s))
            pool = order[:POOL]
            d_best = dedup(pool, toks, meta, "best")
            variants = {"base": order[:K],
                        "dedup[best]": d_best[:K],
                        "dedup[new]": dedup(pool, toks, meta, "new")[:K]}
            for lam in (0.5, 0.7, 0.85):
                variants[f"mmr@{lam}"] = mmr(d_best[:60], s, emb, lam, K)
            for cn in (2, 3):
                variants[f"cap@{cn}"] = cap(d_best, meta, cn, K)
            pool60 = [i for i in d_best if i in set(order[:60])]
            variants["clus@60"] = cluster_pick(pool60, s, emb, K)
            variants["clus@200"] = cluster_pick(d_best, s, emb, K)
            gold_rank = next(r for r, i in enumerate(order, 1) if i in rel)
            for v, lst in variants.items():
                r = next((r for r, i in enumerate(lst, 1) if i in rel), 0)
                rows.append({"id": c["id"], "ticker": t, "category": c.get("category", ""),
                             "variant": v, "hit": 0 < r <= K, "rr": 1 / r if r else 0.0,
                             "gold_rank_dense": gold_rank,
                             "filings": len({meta[i]["acc"] for i in lst}),
                             "items": len({item[ids[i]] for i in lst}),
                             "forms": len({meta[i]["form"] for i in lst})})
        if n % 10 == 0 or n == len(tickers):
            print(f"  {n}/{len(tickers)} tickers", flush=True)

    return report(index, args.split, rows, unscorable, len(cases))


def report(index: Path, split: str, rows, unscorable, n_cases=None) -> int:
    import pandas as pd
    out_dir = Path("out/diversity")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index.name}_{split}"
    if rows is None:
        rows = json.loads((out_dir / f"{stem}_rows.json").read_text())
    else:
        (out_dir / f"{stem}_rows.json").write_text(json.dumps(rows))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="id", columns="variant", values=["hit", "rr"])
    base = piv["hit"]["base"].astype(float)
    gr = df[df.variant == "base"].set_index("id").gold_rank_dense
    band = pd.cut(gr, [0, 8, 20, 60, POOL, 10**9], labels=["1-8", "9-20", "21-60", "61-200", ">200"])
    res = []
    for v in df.variant.unique():
        hit = piv["hit"][v].astype(float)
        dh = (hit - base).to_numpy()
        lo, hi = boot_ci(dh)
        sub = df[df.variant == v]
        row = {"variant": v, "Hit@8": hit.mean(), "MRR@8": piv["rr"][v].mean(),
               "dHit": dh.mean(), "CI95": f"[{lo:+.3f},{hi:+.3f}]",
               "W/L": f"{int((dh > 0).sum())}/{int((dh < 0).sum())}",
               "filings": sub["filings"].mean(), "items": sub["items"].mean()}
        for b in ("9-20", "21-60", "61-200"):
            row[f"rescued {b}"] = int(hit[band == b].sum())
        res.append(row)
    pd.set_option("display.width", 240)
    t = pd.DataFrame(res).set_index("variant").sort_values("Hit@8", ascending=False).round(3)
    txt = (f"{index} split={split}: {n_cases or '?'} questions, scorable {piv.shape[0]}, "
           f"unscorable {len(unscorable)}: {' '.join(unscorable)}\n"
           f"Dense rank of first gold chunk: {band.value_counts().sort_index().to_dict()}\n"
           f"(filings / items = mean distinct filings / section labels in the top {K})\n\n"
           + t.to_string())
    (out_dir / f"{stem}.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
