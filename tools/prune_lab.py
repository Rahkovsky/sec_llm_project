#!/usr/bin/env python3
"""Q2-A (docs/NEXT_STEPS.md): drop the sentences that carry unsupported claims.

The judge already labels every claim of an answer. This estimates, on an existing
run and without any model call, what happens if the sentences behind UNSUPPORTED /
CONTRADICTED claims are removed before the answer is shown:

  * the same rule as verification.prune_unsupported (src/.../generation/prune.py):
    each claim is mapped to its answer sentence by word overlap, and a sentence carrying
    an unsupported or contradicted non-meta claim is removed with any supported claim in
    it (counted as "collateral");
  * the question is relabelled with the report's own rules (tools/provenance_report.py
    questions.csv supplies retrieval recall, wrong-period and the qualifier flags).

    python tools/prune_lab.py out/provenance_16gb_v5 [--samples 20]

Writes <run>/prune/{pruned.json,labels.csv} and prints before/after label counts.
Hand-check the printed sample: a pruned answer must still read as an answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from provenance_report import classify, hardened_hedges

from sec_llm_project.sec_intel.core.types import ClaimVerdict
from sec_llm_project.sec_intel.generation.prune import prune_unsupported
from sec_llm_project.sec_intel.generation.verifier import is_meta_claim

FIELDS = set(ClaimVerdict.__dataclass_fields__)


def prune(answer: dict) -> dict:
    """Pruned text plus claim bookkeeping for one stored answer."""
    claims = [ClaimVerdict(**{k: v for k, v in c.items() if k in FIELDS})
              for c in answer["extra"].get("verification", {}).get("claims", [])]
    for c in claims:  # re-apply the current meta rule, as the report does
        c.meta = c.meta or is_meta_claim(c.claim)
    pr = prune_unsupported(answer["text"], claims)
    return {"text": pr.text, "removed": pr.removed_sentences,
            "claims": [c.to_dict() for c in pr.kept], "collateral": pr.collateral}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--samples", type=int, default=20, help="pruned answers to print")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)
    with open(run / "questions.csv", newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}

    before, after, out, changed = Counter(), Counter(), {}, []
    tot = Counter()
    for qid, r in rows.items():
        if not qid.startswith("finder_"):
            continue
        a = json.loads((run / qid / "answer.json").read_text())
        before[r["error_source"]] += 1
        if a["abstained"]:
            after[r["error_source"]] += 1
            continue
        p = prune(a)
        scored = [c for c in p["claims"] if not (c.get("meta") or is_meta_claim(c["claim"]))]
        ev = r["ev_recall_cover"]
        row = {"answerable": True, "ev_recall": float(ev) if ev not in ("", None) else None,
               "wrong_period": int(r["wrong_period"]), "meta_only": False,
               "unsupported": 0, "contradicted": 0,
               "partial": sum(c["verdict"] == "PARTIAL" for c in scored),
               "hardened": sum(bool(hardened_hedges(c["claim"], c["quote"]))
                               for c in scored if c["verdict"] == "SUPPORTED" and c["quote"]),
               # nothing factual left: the user sees a refusal
               "abstained": not scored}
        label = classify(row)
        after[label] += 1
        if p["removed"]:
            tot["answers_pruned"] += 1
            tot["sentences_removed"] += len(p["removed"])
            tot["collateral_supported_claims"] += p["collateral"]
            tot["emptied"] += row["abstained"]
            out[qid] = {"before": a["text"], "after": p["text"], "removed": p["removed"],
                        "label_before": r["error_source"], "label_after": label}
            changed.append(qid)

    d = run / "prune"
    d.mkdir(exist_ok=True)
    (d / "pruned.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(d / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label_before", "label_after"])
        for qid, o in out.items():
            w.writerow([qid, o["label_before"], o["label_after"]])

    print(dict(tot))
    print(f"{'label':40} before  after")
    for k in sorted(set(before) | set(after), key=lambda k: -before[k]):
        print(f"{k:40} {before[k]:6} {after[k]:6}")
    random.seed(args.seed)
    for qid in random.sample(changed, min(args.samples, len(changed))):
        o = out[qid]
        print(f"\n--- {qid}: {o['label_before']} -> {o['label_after']}")
        print("REMOVED:", " | ".join(o["removed"]))
        print("AFTER:  ", o["after"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
