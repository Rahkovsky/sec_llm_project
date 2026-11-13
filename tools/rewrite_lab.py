#!/usr/bin/env python3
"""Query rewriting for FinDER retrieval: paraphrase, HyDE, multi-query.

FinDER questions are terse analyst phrases ("Cap allocation impact on investor
confidence, MCD"). About a quarter of scorable questions have their gold chunk
ranked below 60 by dense retrieval -- out of reach for any re-ordering (MMR).
This tests whether rewriting the query with the local generator helps.

Phase 1, ``generate`` (Ollama, the production generator; cached, resumable):
  paraphrase  one clear natural question, abbreviations expanded, no new facts
  hyde        a short hypothetical 10-K passage that would answer it (HyDE)
  multi       3 search queries aimed at different parts of a 10-K

Phase 2, ``evaluate`` (no LLM): dense Qwen (instr1) inside the ticker filter,
then MMR (lambda 0.85, pool 60) exactly as production. With several query
vectors, a chunk's relevance is its max cosine over them ("max") or the mean
("mean"). Variants:

  orig            production (original question)
  para            paraphrase only
  hyde_q / hyde_d HyDE text embedded as a query (instr1) / as a document (no instr)
  orig+para       max over {orig, para}
  orig+hyde       max over {orig, hyde_d}
  orig+multi      max over {orig, 3 sub-queries}
  all             max over {orig, para, hyde_d, 3 sub-queries}
  all_mean        same set, mean similarity

    python tools/rewrite_lab.py generate [--split all] [--model gemma4:12b]
    python tools/rewrite_lab.py evaluate [--split dev] [--index data/sec_index_qwen06]
        -> out/rewrite/<index>_<split>.txt, _rows.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from ab_diagnose import QWEN_PROMPTS  # noqa: E402
from ab_finder_dev import boot_ci  # noqa: E402
from diversity_lab import load, mmr  # noqa: E402

OUT = Path("out/rewrite")
K, POOL, LAM = 8, 60, 0.85
INSTR = QWEN_PROMPTS["qwen_instr1"]

SYS = ("You help search a company's SEC 10-K annual report. You never answer from "
       "memory and never add facts that are not in the user's text.")
P_PARA = ("Rewrite this terse analyst search phrase as one clear, complete question about "
          "what the company's 10-K discloses. Expand abbreviations (e.g. 'fin.' -> "
          "'financial', 'cap allocation' -> 'capital allocation', 'mgmt' -> 'management'). "
          "Name the specific disclosure topic. Do not add facts, numbers or a company name "
          "that is not given. Output only the question.\n\n{who}\nPhrase: {q}")
P_HYDE = ("Write a short passage (3-4 sentences) in the formal style of a 10-K annual report "
          "that would answer the question below. Use the wording companies typically use in "
          "the relevant 10-K section (for example Risk Factors, Legal Proceedings, "
          "Cybersecurity, MD&A, or the notes to the financial statements). Do not invent "
          "specific numbers or names; keep it generic. Output only the passage.\n\n"
          "{who}\nQuestion: {q}")
P_MULTI = ("Give exactly 3 different search queries (5-15 words each) for finding the 10-K "
           "passage that answers the question below. Each query should target a different "
           "part of a 10-K where the answer could be disclosed, using the vocabulary a "
           "filing would use, written as a plain phrase without quotation marks. Do not add "
           "facts. Output one query per line, nothing else."
           "\n\n{who}\nQuestion: {q}")
# The entity resolver gives the company in production; without it the model expands
# tickers into words ("ES" -> "Environmental Settlements").
WHO = ("The company is {name} (ticker {ticker}); in the text, {ticker} refers to this company, "
       "not an abbreviation.")


FINDER = "eval/finder_semantic.json"


def cases_for(split: str, path: str = FINDER) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if path != FINDER:  # other sets (e.g. eval/synth_claude.json) have no split
        return cases
    sp = json.loads(Path("eval/finder_split.json").read_text())
    keep = set(sp["dev"]) | set(sp["test"]) if split == "all" else set(sp[split])
    return [c for c in cases if c["id"] in keep]


def cache_path(model: str, blind: bool) -> Path:
    return OUT / f"rewrites_{model.replace(':', '_')}{'_blind' if blind else ''}.json"


def mask_company(q: str, ticker: str, resolver) -> str:
    """Company-blind question: every name/ticker mention -> 'the company'."""
    return resolver.mask(q, [ticker])


def generate(args) -> int:
    from sec_llm_project.sec_intel.llm.base import Message
    from sec_llm_project.sec_intel.llm.providers import OllamaLLM

    llm = OllamaLLM(args.model, temperature=0.0, max_tokens=300, timeout=600, num_ctx=2048)
    OUT.mkdir(parents=True, exist_ok=True)
    path = cache_path(args.model, args.blind)
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [c for c in cases_for(args.split, args.cases) if c["id"] not in cache]
    only = set(args.only.split(",")) if args.only else {"paraphrase", "hyde", "multi"}
    print(f"{len(todo)} questions to rewrite with {args.model} -> {path}", flush=True)

    from sec_llm_project.sec_intel.retrieval.entity import EntityResolver
    names = EntityResolver.from_input_dirs(["data/input/10-K"])  # ticker -> registered name
    company = {t: ph for ph, t in reversed(names._names)}  # longest phrase = full name

    def ask(prompt: str) -> str:
        msgs = [Message("system", SYS), Message("user", prompt)]
        return llm.generate(msgs).text.strip()

    for n, c in enumerate(todo, 1):
        q, t0 = c["question"], time.time()
        rec = {}
        if args.blind:  # no company name in the prompt, none left in the question
            q, who = mask_company(q, c["ticker"], names), ""
            rec["masked_question"] = q
        else:
            who = WHO.format(name=company.get(c["ticker"], c["ticker"]).title(),
                             ticker=c["ticker"])
        if "paraphrase" in only:
            rec["paraphrase"] = ask(P_PARA.format(q=q, who=who))
        if "hyde" in only:
            rec["hyde"] = ask(P_HYDE.format(q=q, who=who))
        if "multi" in only:
            lines = [ln.strip(" -*0123456789.\"") for ln in ask(P_MULTI.format(q=q, who=who))
                     .splitlines()]
            rec["multi"] = [ln for ln in lines if len(ln.split()) >= 3][:3]
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
    rw = json.loads(cache_path(args.model, False).read_text())
    bp = cache_path(args.model, True)
    blind = json.loads(bp.read_text()) if bp.exists() else {}
    cases = [c for c in cases_for(args.split, args.cases) if c["id"] in rw]
    tickers = sorted({c["ticker"] for c in cases})
    print(f"{index} {args.split}: {len(cases)} questions with rewrites", flush=True)
    chunks = load(index, set(tickers))
    col = chromadb.PersistentClient(path=str(index / "chroma_db")).get_collection("sec_filings")
    qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", model_kwargs={"dtype": torch.float32})

    def enc(texts: list[str]) -> np.ndarray:
        return qwen.encode(texts, normalize_embeddings=True)

    by_t = defaultdict(list)
    for c in cases:
        by_t[c["ticker"]].append(c)
    rows = []
    for n, t in enumerate(tickers, 1):
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
            r = rw[c["id"]]
            v_orig, v_hyde_q = enc([INSTR + c["question"], INSTR + r["hyde"]])
            sets = {"orig": [v_orig], "hyde_q": [v_hyde_q], "orig+hyde_q": [v_orig, v_hyde_q]}
            means = {"orig+hyde_q_mean": "orig+hyde_q"}
            if "paraphrase" in r and "multi" in r:
                multi = [m for m in r["multi"] if m.strip()]
                qv = enc([INSTR + x for x in [r["paraphrase"], *multi]])
                v_para, v_multi = qv[0], qv[1:]
                v_hyde_d = enc([r["hyde"]])[0]
                sets.update({
                    "para": [v_para], "hyde_d": [v_hyde_d],
                    "orig+para": [v_orig, v_para], "orig+hyde": [v_orig, v_hyde_d],
                    "orig+multi": [v_orig, *v_multi],
                    "all": [v_orig, v_para, v_hyde_d, *v_multi],
                    "all_q": [v_orig, v_para, v_hyde_q, *v_multi],
                    "orig+multi+hyde_q": [v_orig, v_hyde_q, *v_multi]})
                means.update({"all_mean": "all", "all_q_mean": "all_q",
                              "orig+multi_mean": "orig+multi",
                              "orig+multi+hyde_q_mean": "orig+multi+hyde_q"})
            if c["id"] in blind:
                v_hb = enc([INSTR + blind[c["id"]]["hyde"]])[0]
                sets.update({"hyde_blind": [v_hb], "orig+hyde_blind": [v_orig, v_hb]})
                means["orig+hyde_blind_mean"] = "orig+hyde_blind"
            orig_rank = None
            for name, vs in [*sets.items(), *((m, sets[src]) for m, src in means.items())]:
                sims = emb @ np.stack(vs).T
                s = sims.mean(axis=1) if name in means else sims.max(axis=1)
                order = list(np.argsort(-s))
                if name == "orig":
                    orig_rank = next(k for k, i in enumerate(order, 1) if i in rel)
                lst = mmr(order[:POOL], s, emb, LAM, K)
                rank = next((k for k, i in enumerate(lst, 1) if i in rel), 0)
                pool_rank = next(k for k, i in enumerate(order, 1) if i in rel)
                rows.append({"id": c["id"], "ticker": t, "category": c.get("category", ""),
                             "variant": name, "hit": 0 < rank <= K,
                             "rr": 1 / rank if rank else 0.0, "gold_rank": pool_rank,
                             "orig_rank": orig_rank})
        if n % 10 == 0 or n == len(tickers):
            print(f"  {n}/{len(tickers)} tickers", flush=True)

    stem = f"{index.name}_{Path(args.cases).stem}_{args.split}"
    (OUT / f"{stem}_rows.json").write_text(json.dumps(rows))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="id", columns="variant", values=["hit", "rr", "gold_rank"])
    base = piv["hit"]["orig"].astype(float)
    orank = piv["gold_rank"]["orig"]
    deep = orank > POOL
    res = []
    for v in df.variant.unique():
        hit = piv["hit"][v].astype(float)
        dh = (hit - base).to_numpy()
        lo, hi = boot_ci(dh)
        res.append({"variant": v, "Hit@8": hit.mean(), "MRR@8": piv["rr"][v].mean(),
                    "dHit": dh.mean(), "CI95": f"[{lo:+.3f},{hi:+.3f}]",
                    "W/L": f"{int((dh > 0).sum())}/{int((dh < 0).sum())}",
                    "Hit@8 deep": hit[deep].mean(),
                    "median gold rank": float(piv["gold_rank"][v].median())})
    pd.set_option("display.width", 240)
    t = pd.DataFrame(res).set_index("variant").sort_values("Hit@8", ascending=False).round(3)
    secs = [rw[i].get("seconds", 0) for i in piv.index]
    txt = (f"{index} split={args.split}: scorable {len(piv)}; all variants use MMR "
           f"(lambda {LAM}, pool {POOL}); Delta vs orig (= production)\n"
           f"'deep' = gold ranked > {POOL} for the original question: {int(deep.sum())} questions\n"
           f"rewrite time per question ({args.model}, 3 calls): median {np.median(secs):.1f}s\n\n"
           + t.to_string())
    (OUT / f"{stem}.txt").write_text(txt)
    print(txt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["generate", "evaluate"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--index", default="data/sec_index_qwen06")
    ap.add_argument("--cases", default=FINDER)
    ap.add_argument("--blind", action="store_true",
                    help="generate: company-blind rewrites (separate cache)")
    ap.add_argument("--only", help="generate: comma list of paraphrase,hyde,multi")
    args = ap.parse_args()
    args.split = args.split or ("all" if args.phase == "generate" else "dev")
    return generate(args) if args.phase == "generate" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
