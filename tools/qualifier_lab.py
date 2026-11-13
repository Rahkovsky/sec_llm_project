#!/usr/bin/env python3
"""Dropped-qualifier detection: score the judge-model check against hand labels.

eval/qualifier_labels.json holds the 73 claims the lexical proxy flagged in the
v3 end-to-end run, hand-labelled (real = the claim states as fact what the
filing's hedge qualifies). This runs generation.qualifiers.QualifierChecker with
the config's judge model, one call per answer (as in production), and prints
precision/recall of the lexical proxy (by construction: every item) and of the
judge.

    python tools/qualifier_lab.py --config config/local_16gb_v4.yaml
    python tools/qualifier_lab.py --subset-out eval/qualifier_subset.json   # cases to regenerate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.generation.qualifiers import (
    QualifierChecker,
    QualifierItem,
)
from sec_llm_project.sec_intel.llm.factory import build_llm
from sec_llm_project.utils.env_config import load_environment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/local_16gb_v4.yaml")
    ap.add_argument("--labels", default="eval/qualifier_labels.json")
    ap.add_argument("--cases", default="eval/gold_plus_placebo.json")
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--subset-out", help="write the cases whose v3 answer had a candidate")
    args = ap.parse_args()
    items = json.loads(Path(args.labels).read_text())["items"]

    if args.subset_out:
        ids = sorted({it["id"] for it in items})
        cases = [c for c in json.loads(Path(args.cases).read_text()) if c["id"] in ids]
        Path(args.subset_out).write_text(json.dumps(cases, indent=1))
        print(f"wrote {len(cases)} cases -> {args.subset_out}")
        return 0

    load_environment()
    cfg = AppConfig.load(args.config)
    checker = QualifierChecker(build_llm(cfg.verification.judge), prompt=args.prompt)
    by_answer: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_answer[it["id"]].append(i)
    pred: dict[int, bool | None] = {}
    for idx in by_answer.values():
        got = checker.check([QualifierItem(items[i]["claim"], items[i]["evidence"]) for i in idx])
        pred.update(zip(idx, got, strict=False))

    real = sum(it["real"] for it in items)
    tp = sum(1 for i, it in enumerate(items) if pred[i] and it["real"])
    fp = sum(1 for i, it in enumerate(items) if pred[i] and not it["real"])
    unk = sum(1 for v in pred.values() if v is None)
    print(f"items {len(items)}, real {real}; judge {cfg.verification.judge.model} "
          f"prompt {args.prompt}, "
          f"{len(by_answer)} calls, unparsed {unk}")
    print(f"lexical proxy: precision {real / len(items):.2f}  recall 1.00 (by construction)")
    print(f"judge check:   precision {tp / max(1, tp + fp):.2f}  recall {tp / max(1, real):.2f}  "
          f"(tp {tp}, fp {fp}, fn {real - tp})")
    for i, it in enumerate(items):
        if bool(pred[i]) != it["real"]:
            kind = "FN" if it["real"] else "FP"
            print(f"  {kind} {it['id']}: {it['claim'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
