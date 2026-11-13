#!/usr/bin/env python3
"""Judge-checked dropped qualifiers for finished runs (same checker for every run).

The lexical proxy (precision 0.19 on hand labels) is too noisy to compare runs, and
runs made before the checker existed carry no judged count. This collects the
lexical candidates (claim + supporting quote, SUPPORTED/PARTIAL) of every answered
question in each run directory and labels them with QualifierChecker (prompt v2,
one judge call per answer), restricted to the questions of --cases.

    python tools/qualifier_rescore.py --cases eval/qualifier_subset.json \
        out/provenance_16gb_v3 out/provenance_16gb_v3_gold out/provenance_16gb_v4_qualsub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.generation.qualifiers import (
    QualifierChecker,
    QualifierItem,
    is_candidate,
)
from sec_llm_project.sec_intel.generation.verifier import is_meta_claim
from sec_llm_project.sec_intel.llm.factory import build_llm
from sec_llm_project.utils.env_config import load_environment


def candidates(answer: dict) -> list[QualifierItem]:
    out = []
    for c in answer.get("extra", {}).get("verification", {}).get("claims", []):
        if (c.get("verdict") in ("SUPPORTED", "PARTIAL") and c.get("quote")
                and not is_meta_claim(c["claim"]) and is_candidate(c["claim"], c["quote"])):
            out.append(QualifierItem(c["claim"], c["quote"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cases", default="eval/qualifier_subset.json")
    ap.add_argument("--config", default="config/local_16gb_v4.yaml")
    args = ap.parse_args()
    load_environment()
    cfg = AppConfig.load(args.config)
    checker = QualifierChecker(build_llm(cfg.verification.judge), prompt="v2")
    ids = [c["id"] for c in json.loads(Path(args.cases).read_text())]
    result = {}
    for run in args.runs:
        per = {}
        for i in ids:
            p = Path(run) / i / "answer.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text())
            if a.get("abstained"):
                continue
            items = candidates(a)
            got = checker.check(items) if items else []
            per[i] = {"candidates": len(items), "dropped": sum(bool(g) for g in got),
                      "claims": [it.claim for it, g in zip(items, got, strict=False) if g]}
        result[run] = per
        n_ans = len(per)
        cand = sum(v["candidates"] for v in per.values())
        drop = sum(v["dropped"] for v in per.values())
        print(f"{run}: answered {n_ans}, lexical candidates {cand}, judged dropped {drop} "
              f"in {sum(v['dropped'] > 0 for v in per.values())} answers", flush=True)
    Path("out/qualifier_rescore.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
