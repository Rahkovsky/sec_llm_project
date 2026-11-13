#!/usr/bin/env python3
"""Accuracy of the deterministic entity resolver against FinDER's ticker labels.

    python tools/eval_entity.py [--cases eval/gold_plus_placebo.json]
      -> out/entity_eval.csv + summary (exact / includes gold / wrong / none)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.retrieval.entity import EntityResolver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/gold_plus_placebo.json")
    ap.add_argument("--out", default="out/entity_eval.csv")
    args = ap.parse_args()
    r = EntityResolver.from_input_dirs(["data/input/10-K", "data/input/10-Q"])
    data = json.loads(Path(args.cases).read_text())
    cases = data if isinstance(data, list) else data.get("cases", data)
    rows, tally = [], Counter()
    for c in cases:
        gold = (c.get("ticker") or "").upper()
        res = r.resolve(c["question"])
        if not gold:
            outcome = "no_gold"
        elif res.tickers == [gold]:
            outcome = "exact"
        elif gold in res.tickers:
            outcome = "includes_gold"
        elif res.tickers:
            outcome = "wrong"
        else:
            outcome = "none"
        tally[outcome] += 1
        rows.append({"id": c["id"], "gold": gold, "resolved": " ".join(res.tickers),
                     "method": res.method, "outcome": outcome,
                     "evidence": json.dumps(res.evidence), "question": c["question"]})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = sum(v for k, v in tally.items() if k != "no_gold")
    print(f"{len(rows)} questions ({n} with a gold ticker) -> {args.out}")
    for k in ("exact", "includes_gold", "wrong", "none", "no_gold"):
        if tally[k]:
            print(f"  {k:14s} {tally[k]:4d}" + (f"  ({tally[k] / n:.1%})" if k != "no_gold" else ""))
    print("\nNot exact:")
    for row in rows:
        if row["outcome"] not in ("exact",):
            print(f"  {row['id']} gold={row['gold'] or '-'} got={row['resolved'] or '-'} "
                  f"{row['evidence']} | {row['question'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
