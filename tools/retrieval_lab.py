#!/usr/bin/env python3
"""Retrieval experiment harness: compare query/ranking variants on a fixed split.

Every variant is a composition of optional steps over dense retrieval:

    query text  -> clean (ticker -> company name, strip placeholders)
                -> instruction (Qwen3-Embedding query instruction variant)
                -> hyde (LLM writes a hypothetical 10-K passage; embedded as a doc)
    dense top-N (ticker filter)
                -> boost (soft section bonus from a keyword router)
                -> dedupe (drop near-duplicate chunks, e.g. same text across years)
                -> rerank (cross-encoder over the top `depth` candidates)
    top-k       -> Hit@k / MRR under the strict and cover relevance rules

To avoid tuning on the test set, questions are split once (stratified by
category, fixed seed) into dev/test, saved to eval/finder_split.json. Iterate
with --split dev; score the final choice once with --split test.

    python tools/retrieval_lab.py --config config/local_16gb.yaml --split dev \
        --variants base clean instr1 clean+instr1+rerank_qwen
    python tools/retrieval_lab.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit

SPLIT_PATH = Path("eval/finder_split.json")
RULES = {"strict": None, "cover": CONTAINMENT_THRESHOLD}

INSTRUCTIONS = {
    "instr0": "Instruct: Given a question about SEC filings, retrieve the passages that "
              "answer it\nQuery: ",  # production default (huggingface.py)
    "instr1": "Instruct: Given a terse search phrase from a financial analyst about a "
              "company, retrieve the passage of the company's 10-K annual report that "
              "answers it\nQuery: ",
    "instr2": "Instruct: Given a search query about a public company's disclosures, "
              "retrieve relevant passages from its SEC annual report\nQuery: ",
    "noinstr": "",
}

# Keyword router -> 10-K items (soft boost only). Derived from where gold text
# lives: cyber -> 1C; litigation/contingencies -> 3 + financial-statement notes
# (8); business/strategy -> 1; executives/board -> 10 + "Executive Officers"
# (after Item 4) + signatures (after Item 16).
ROUTES = [
    (r"cyber|information security|data breach|ransomware|hack", {"1C"}),
    (r"legal|litigation|lawsuit|proceeding|contingen|settlement|lawsuits|liabilit|"
     r"regulator|investigation|claims", {"3", "8"}),
    (r"board|director|executive|officer|governance|compensation|ceo|cfo|leadership|"
     r"management team", {"10", "4", "16"}),
    (r"business|segment|product|customer|competit|strategy|overview|operations|market|"
     r"revenue|employees|human capital", {"1"}),
]

HYDE_PROMPT = ("Write one paragraph (80-120 words) exactly as it would appear in "
               "{company}'s most recent Form 10-K that addresses the following topic. "
               "Use formal SEC-filing language. Do not add a title.\n\nTopic: {query}")


# ----------------------------------------------------------------- helpers
def make_split(cases: list[dict], seed: int = 7) -> dict[str, list[str]]:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c["id"])
    rng = random.Random(seed)
    dev, test = [], []
    for cat_ids in by_cat.values():
        ids = sorted(cat_ids)
        rng.shuffle(ids)
        half = len(ids) // 2
        dev += ids[:half]
        test += ids[half:]
    return {"seed": seed, "dev": sorted(dev), "test": sorted(test)}


def route(query: str) -> set[str]:
    q = query.lower()
    items: set[str] = set()
    for pat, its in ROUTES:
        if re.search(pat, q):
            items |= its
    return items


def toks(text: str) -> set[str]:
    return set(text.lower().split())


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class Lab:
    def __init__(self, config: str, depth: int) -> None:
        from sec_llm_project.sec_intel.core.config import AppConfig
        from sec_llm_project.sec_intel.pipeline import SECIntelPipeline

        self.cfg = AppConfig.load(config)
        self.cfg.retrieval.use_bm25 = False
        self.pipe = SECIntelPipeline(self.cfg)
        self.pipe.load_index()
        self.index, self.emb = self.pipe.index, self.pipe.embedder
        self.depth = depth
        names: dict[str, Counter[str]] = defaultdict(Counter)
        for ch in self.index.all_chunks():
            names[ch.metadata.ticker][ch.metadata.company] += 1
        self.company = {t: self._pretty(c.most_common(1)[0][0]) for t, c in names.items()}
        self._dense_cache: dict[tuple[str, str], list[tuple[str, float]]] = {}
        self._hyde_cache_path = Path("out/retrieval_lab/hyde_cache.json")
        self._hyde = (json.loads(self._hyde_cache_path.read_text())
                      if self._hyde_cache_path.exists() else {})
        self._rerankers: dict[str, object] = {}
        self._rerank_cache: dict[tuple[str, str, str], float] = {}

    @staticmethod
    def _pretty(name: str) -> str:
        name = re.sub(r"[,.]?\s+(INC|CORP|CORPORATION|CO|LTD|PLC|LLC|N\.V|HOLDINGS)\.?$", "",
                      name.strip(), flags=re.I)
        return name.title() if name.isupper() else name

    # ---- query side
    def clean(self, q: str, ticker: str) -> str:
        comp = self.company.get(ticker, ticker)
        q = re.sub(r"\(\s*ticker\s*\)|\bTICKER\b", comp, q)
        return re.sub(rf"\b{re.escape(ticker)}\b", f"{comp} ({ticker})", q, count=1)

    def hyde(self, q: str, ticker: str) -> str:
        key = f"{ticker}|{q}"
        if key not in self._hyde:
            from sec_llm_project.sec_intel.llm.base import Message
            from sec_llm_project.sec_intel.llm.factory import build_llm
            if not hasattr(self, "_llm"):
                self._llm = build_llm(self.cfg.llm)
            prompt = HYDE_PROMPT.format(company=self.company.get(ticker, ticker), query=q)
            self._hyde[key] = self._llm.generate([Message("user", prompt)],
                                                 temperature=0.0, max_tokens=256).text.strip()
            self._hyde_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._hyde_cache_path.write_text(json.dumps(self._hyde))
        return self._hyde[key]

    def dense(self, text: str, instr: str, ticker: str) -> list[tuple[str, float]]:
        key = (instr + text, ticker)
        if key not in self._dense_cache:
            vec = self.emb._encode([instr + text])[0]
            self._dense_cache[key] = self.index.dense_search(vec, self.depth, {"ticker": ticker})
        return self._dense_cache[key]

    # ---- ranking side
    def boost(self, ranked: list[tuple[str, float]], q: str, bonus: float):
        items = route(q)
        if not items:
            return ranked
        out = [(cid, s + (bonus if self.index.get(cid).item_number in items else 0.0))
               for cid, s in ranked]
        return sorted(out, key=lambda t: t[1], reverse=True)

    def dedupe(self, ranked: list[tuple[str, float]], thr: float = 0.85):
        kept: list[tuple[str, float, set[str]]] = []
        for cid, s in ranked:
            t = toks(self.index.get(cid).text)
            if all(jaccard(t, kt) < thr for _, _, kt in kept):
                kept.append((cid, s, t))
        return [(c, s) for c, s, _ in kept]

    def rerank(self, ranked: list[tuple[str, float]], q: str, model: str, depth: int):
        pool = ranked[:depth]
        need = [cid for cid, _ in pool if (model, q, cid) not in self._rerank_cache]
        if need:
            scores = self._reranker(model)(q, [self.index.get(c).text for c in need])
            for cid, s in zip(need, scores, strict=True):
                self._rerank_cache[(model, q, cid)] = s
        head = sorted(((cid, self._rerank_cache[(model, q, cid)]) for cid, _ in pool),
                      key=lambda t: t[1], reverse=True)
        return head + ranked[depth:]

    def _reranker(self, model: str):
        if model not in self._rerankers:
            self._rerankers[model] = (QwenReranker() if model == "qwen"
                                      else CrossEncoderScorer(model))
        return self._rerankers[model]


CROSS_ENCODERS = {"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                  "bge": "BAAI/bge-reranker-v2-m3"}


class CrossEncoderScorer:
    def __init__(self, name: str) -> None:
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(CROSS_ENCODERS[name], max_length=512)

    def __call__(self, q: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.m.predict([(q, d) for d in docs], batch_size=16)]


class QwenReranker:
    """Qwen3-Reranker-0.6B, scored as P(yes) per the model card."""

    NAME = "Qwen/Qwen3-Reranker-0.6B"
    INSTRUCT = ("Given a search query from a financial analyst about a company, judge "
                "whether this passage from the company's SEC filing answers it")

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.NAME, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.NAME, torch_dtype=torch.float16).to(self.dev).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(
            "<|im_start|>system\nJudge whether the Document meets the requirements based on "
            "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
            "\"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
        self.suf = self.tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                   add_special_tokens=False)
        self.max_len = 1024

    def __call__(self, q: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), 8):
            texts = [f"<Instruct>: {self.INSTRUCT}\n<Query>: {q}\n<Document>: {d}"
                     for d in docs[i:i + 8]]
            enc = self.tok(texts, padding=False, truncation="longest_first",
                           return_attention_mask=False,
                           max_length=self.max_len - len(self.pre) - len(self.suf))
            enc["input_ids"] = [self.pre + ids + self.suf for ids in enc["input_ids"]]
            batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                # Only the last position is scored; logits over all positions x
                # 152k vocab OOMs the GPU on 16 GB.
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out += self.torch.log_softmax(pair, dim=1)[:, 1].exp().tolist()
        return out


# ----------------------------------------------------------------- variants
def parse_variant(spec: str) -> dict:
    """'clean+instr1+boost05+dedupe+rerank_qwen40' -> step settings."""
    v = {"clean": False, "instr": "instr0", "hyde": None, "boost": 0.0, "dedupe": False,
         "rerank": None, "rdepth": 40}
    for part in spec.split("+"):
        if part == "base":
            continue
        if part == "clean":
            v["clean"] = True
        elif part in INSTRUCTIONS:
            v["instr"] = part
        elif part in ("hyde", "hydemix"):
            v["hyde"] = part
        elif m := re.fullmatch(r"boost(\d+)", part):
            v["boost"] = int(m.group(1)) / 100
        elif part == "dedupe":
            v["dedupe"] = True
        elif m := re.fullmatch(r"rerank_(qwen|minilm|bge)(\d*)", part):
            v["rerank"], v["rdepth"] = m.group(1), int(m.group(2) or 40)
        else:
            raise SystemExit(f"unknown variant step: {part}")
    return v


def run_variant(lab: Lab, spec: str, cases: list[dict], k: int):
    v = parse_variant(spec)
    tops = {}
    for c in cases:
        t = c["ticker"]
        q = lab.clean(c["question"], t) if v["clean"] else c["question"]
        instr = INSTRUCTIONS[v["instr"]]
        if v["hyde"] == "hyde":
            ranked = lab.dense(lab.hyde(q, t), "", t)  # hypothetical passage = a document
        elif v["hyde"] == "hydemix":  # RRF of query and HyDE lists
            a, b = lab.dense(q, instr, t), lab.dense(lab.hyde(q, t), "", t)
            sc: dict[str, float] = defaultdict(float)
            for lst in (a, b):
                for r, (cid, _) in enumerate(lst):
                    sc[cid] += 1 / (60 + r)
            ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = lab.dense(q, instr, t)
        if v["boost"]:
            ranked = lab.boost(ranked, c["question"], v["boost"])
        if v["dedupe"]:
            ranked = lab.dedupe(ranked)
        if v["rerank"]:
            ranked = lab.rerank(ranked, q, v["rerank"], v["rdepth"])
        tops[c["id"]] = [cid for cid, _ in ranked[:k]]
    return tops


def score(lab: Lab, tops: dict[str, list[str]], cases: list[dict], rel_cache: dict):
    res = {r: {"hit": [], "rr": []} for r in RULES}
    per_q = {}
    for c in cases:
        ids = tops[c["id"]]
        for rule, cont in RULES.items():
            first = 0
            for i, cid in enumerate(ids, 1):
                key = (c["id"], cid, rule)
                if key not in rel_cache:
                    rel_cache[key] = passage_hit(lab.index.get(cid).text,
                                                 c["relevant_passages"], 0.5, cont)
                if rel_cache[key]:
                    first = i
                    break
            res[rule]["hit"].append(float(first > 0))
            res[rule]["rr"].append(1 / first if first else 0.0)
            if rule == "cover":
                per_q[c["id"]] = first
    return {r: (sum(x["hit"]) / len(x["hit"]), sum(x["rr"]) / len(x["rr"]))
            for r, x in res.items()}, per_q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/local_16gb.yaml")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--variants", nargs="*", default=["base"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=100, help="dense candidates per query")
    ap.add_argument("--out", default="out/retrieval_lab")
    ap.add_argument("--list", action="store_true", help="print variant steps and exit")
    args = ap.parse_args()
    if args.list:
        print("steps: base clean", " ".join(INSTRUCTIONS), "hyde hydemix boostNN dedupe "
              "rerank_{qwen,minilm,bge}[depth]  (join with +)")
        return 0

    all_cases = json.loads(Path(args.cases).read_text())
    if not SPLIT_PATH.exists():
        SPLIT_PATH.write_text(json.dumps(make_split(all_cases), indent=1))
        print(f"wrote {SPLIT_PATH}")
    split = json.loads(SPLIT_PATH.read_text())
    keep = set(split["dev"] + split["test"]) if args.split == "all" else set(split[args.split])
    cases = [c for c in all_cases if c["id"] in keep]
    cats = Counter(c["category"] for c in cases)
    print(f"{args.split}: {len(cases)} questions {dict(cats)}")

    t0 = time.perf_counter()
    lab = Lab(args.config, args.depth)
    print(f"index loaded in {time.perf_counter() - t0:.0f}s\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rel_cache: dict = {}
    hdr = f"{'variant':44s} {'Hit@k cover':>11} {'MRR':>6} {'Hit@k strict':>12} {'MRR':>6} {'s/q':>6}"
    print(hdr + "\n" + "-" * len(hdr))
    summary = []
    for spec in args.variants:
        t = time.perf_counter()
        tops = run_variant(lab, spec, cases, args.k)
        dt = (time.perf_counter() - t) / len(cases)
        m, per_q = score(lab, tops, cases, rel_cache)
        print(f"{spec:44s} {m['cover'][0]:>11.3f} {m['cover'][1]:>6.3f} "
              f"{m['strict'][0]:>12.3f} {m['strict'][1]:>6.3f} {dt:>6.2f}", flush=True)
        bycat = {cat: sum(per_q[c["id"]] > 0 for c in cases if c["category"] == cat) / n
                 for cat, n in cats.items()}
        summary.append({"variant": spec, "split": args.split, "n": len(cases),
                        "hit_cover": m["cover"][0], "mrr_cover": m["cover"][1],
                        "hit_strict": m["strict"][0], "mrr_strict": m["strict"][1],
                        "sec_per_q": dt, **{f"hit_cover_{k}": v for k, v in bycat.items()}})
        (out / f"perq_{args.split}_{spec}.json").write_text(json.dumps(per_q))
    path = out / f"summary_{args.split}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=summary[0].keys())
        if new:
            w.writeheader()
        w.writerows(summary)
    print(f"\nappended {len(summary)} rows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
