#!/usr/bin/env python3
"""Is each eval question's gold passage findable in the index at all?

Retrieval is scored as a hit when a retrieved chunk contains >= 50% of a gold
passage's tokens (token recall). This tool streams chunks.jsonl once and, for
every question, finds the best-matching chunk of that ticker ANYWHERE in the
index — the ceiling any retriever could reach. A question whose best chunk is
below the threshold is an indexing/corpus failure, not a ranking failure.

Model-free. Writes a per-question CSV (best recall, best chunk id, its filing).

    python tools/gold_coverage.py --config config/local_16gb.yaml \
        --cases eval/finder_semantic.json --out out/gold_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.core.config import AppConfig

THRESHOLD = 0.5  # same as tools/tune_weights.py


def toks(text: str) -> set[str]:
    return set(text.lower().split())  # same tokenisation as the hit metric


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/local_16gb.yaml")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--out", default="out/gold_coverage.csv")
    args = ap.parse_args()

    cases = [c for c in json.loads(Path(args.cases).read_text()) if c.get("relevant_passages")]
    by_ticker: dict[str, list[tuple[str, int, set[str]]]] = defaultdict(list)
    for c in cases:
        for j, p in enumerate(c["relevant_passages"]):
            by_ticker[c["ticker"]].append((c["id"], j, toks(p)))

    # best[(qid, passage_idx)] = (recall, chunk_id, filing_type, filing_date, item)
    best: dict[tuple[str, int], tuple[float, str, str, str, str]] = {}
    n_ticker_chunks: dict[str, int] = defaultdict(int)
    catalog = Path(AppConfig.from_file(args.config).index.path) / "chunks.jsonl"
    with open(catalog, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            m = d.get("metadata", {})
            t = m.get("ticker")
            if t not in by_ticker:
                continue
            n_ticker_chunks[t] += 1
            ct = toks(d.get("text", ""))
            for qid, j, pt in by_ticker[t]:
                if not pt:
                    continue
                r = len(pt & ct) / len(pt)
                if r > best.get((qid, j), (-1.0,))[0]:
                    best[(qid, j)] = (r, d["chunk_id"], m.get("filing_type", ""),
                                      m.get("filing_date", ""), d.get("item_number", ""))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases:
        per = [best.get((c["id"], j), (0.0, "", "", "", ""))
               for j in range(len(c["relevant_passages"]))]
        top = max(per, key=lambda x: x[0])
        plen = max(len(p) for p in c["relevant_passages"])
        rows.append({"id": c["id"], "ticker": c["ticker"], "category": c["category"],
                     "n_passages": len(per), "max_passage_chars": plen,
                     "ticker_chunks": n_ticker_chunks.get(c["ticker"], 0),
                     "best_recall": round(top[0], 4), "best_chunk": top[1],
                     "best_form": top[2], "best_date": top[3], "best_item": top[4]})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    findable = sum(r["best_recall"] >= THRESHOLD for r in rows)
    print(f"{len(rows)} questions; gold findable (best chunk >= {THRESHOLD:.0%} token recall): "
          f"{findable} ({findable / len(rows):.1%})")
    for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]:
        n = sum(lo <= r["best_recall"] < hi for r in rows)
        print(f"  best recall {lo:.1f}-{min(hi, 1.0):.1f}: {n}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
