#!/usr/bin/env python3
"""How much of each FinDER question's gold evidence is in our index, and where?

For every question and every filing of its ticker, measures what fraction of
the gold passage's distinct tokens is covered by that filing's matching chunks
(a chunk matches when it holds >= 50% of the passage, or lies >= 80% inside it
with >= 40 tokens: the "cover" rule of evaluation/metrics.py). The best filing
per question tells whether we hold the source text, and in which report.

Model-free, streams chunks.jsonl once.

    python tools/evidence_audit.py [INDEX_DIR [OUT_CSV]]   # -> out/evidence_audit.csv + summary
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.evaluation.metrics import (  # noqa: E402
    CONTAINMENT_THRESHOLD,
    MIN_CONTAINMENT_TOKENS,
)

INDEX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sec_index_qwen06")


def main() -> int:
    cases = json.loads(Path("eval/finder_semantic.json").read_text())
    split = json.loads(Path("eval/finder_split.json").read_text())
    dev = set(split["dev"])
    by_t = defaultdict(list)
    for c in cases:
        c["ptoks"] = [set(p.lower().split()) for p in c["relevant_passages"]]
        by_t[c["ticker"]].append(c)
    # (qid, accession) -> covered passage tokens per passage; plus filing info
    cov: dict[tuple[str, str], list[set[str]]] = {}
    best_chunk: dict[str, float] = defaultdict(float)
    filing: dict[str, tuple[str, str, str]] = {}
    filings_per_ticker: dict[str, Counter] = defaultdict(Counter)
    seen_acc: set[str] = set()
    with open(INDEX / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            m = d["metadata"]
            t = m.get("ticker")
            if t not in by_t:
                continue
            acc = m.get("accession", "")
            if acc not in seen_acc:
                seen_acc.add(acc)
                filings_per_ticker[t][m.get("filing_type", "?")] += 1
                filing[acc] = (m.get("filing_type", "?"), m.get("fiscal_year", "?"),
                               m.get("filing_date", "?"))
            ct = set(d["text"].lower().split())
            for c in by_t[t]:
                for i, pt in enumerate(c["ptoks"]):
                    inter = ct & pt
                    rec = len(inter) / len(pt) if pt else 0.0
                    best_chunk[c["id"]] = max(best_chunk[c["id"]], rec)
                    inside = len(inter) / len(ct) if ct else 0.0
                    if rec >= 0.5 or (len(ct) >= MIN_CONTAINMENT_TOKENS
                                      and inside >= CONTAINMENT_THRESHOLD):
                        key = (c["id"], acc)
                        if key not in cov:
                            cov[key] = [set() for _ in c["ptoks"]]
                        cov[key][i] |= inter
    rows = []
    for c in cases:
        best = (0.0, "", ("-", "-", "-"))
        for (qid, acc), sets in cov.items():
            if qid != c["id"]:
                continue
            frac = max(len(s) / len(pt) for s, pt in zip(sets, c["ptoks"], strict=True))
            if frac > best[0]:
                best = (frac, acc, filing[acc])
        n_filings = sum(1 for (qid, _a) in cov if qid == c["id"])
        f = filings_per_ticker[c["ticker"]]
        rows.append({"id": c["id"], "split": "dev" if c["id"] in dev else "test",
                     "ticker": c["ticker"], "category": c["category"],
                     "passage_chars": max(len(p) for p in c["relevant_passages"]),
                     "best_coverage": round(best[0], 3), "best_filing": best[1],
                     "best_form": best[2][0], "best_fy": best[2][1], "best_date": best[2][2],
                     "filings_with_match": n_filings,
                     "best_single_chunk": round(best_chunk[c["id"]], 3),
                     "ticker_10k": f.get("10-K", 0), "ticker_10q": f.get("10-Q", 0)})
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/evidence_audit.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    bins = pd.cut(df.best_coverage, [-0.01, 0.0, 0.5, 0.9, 1.0],
                  labels=["none (0)", "fragment (<50%)", "most (50-90%)", "all (>=90%)"])
    print(f"wrote {out}\n")
    print("Filings per FinDER ticker:", df.drop_duplicates("ticker")[["ticker_10k", "ticker_10q"]]
          .agg(["min", "median", "max"]).to_dict())
    print("\nShare of the gold passage our best filing contains:")
    print(pd.crosstab(bins, df.split, margins=True).to_string())
    print("\nBest-matching filing, form x fiscal year (questions with any match):")
    hit = df[df.best_coverage > 0]
    print(pd.crosstab(hit.best_form, hit.best_fy, margins=True).to_string())
    print("\nMedian coverage by category:", df.groupby("category").best_coverage.median().round(2)
          .to_dict())
    print("\nFilings containing the evidence per question (repeats across years):",
          hit.filings_with_match.describe()[["min", "50%", "max"]].to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
