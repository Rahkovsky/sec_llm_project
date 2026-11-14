#!/usr/bin/env python3
"""Re-apply the verifier's quote and figure checks to a stored run (EXPERIMENTS §20.3).

The judge's own verdict is kept; only the deterministic invariants are re-run with
the current ``quote_matches`` (citation re-pointing included). Runs made before the
``judge_verdict`` field existed did not store the judge's verdict for a claim the quote
check downgraded; such a claim (UNSUPPORTED with a quote and a citation) is taken as
SUPPORTED by the judge, an upper bound since some were PARTIAL.

    python tools/quote_rescore.py out/provenance_16gb_v5 --config config/local_16gb_v5.yaml

Prints the label counts before, after re-scoring, and after re-scoring + pruning, and
writes <run>/quote_rescore.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from judge_lab import _answers, _context, label
from provenance_report import chunk_texts

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.core.types import ClaimVerdict
from sec_llm_project.sec_intel.generation.verifier import CitationVerifier

FIELDS = set(ClaimVerdict.__dataclass_fields__)


def rescore(claims: list[dict], context) -> list[dict]:
    out = []
    for c in claims:
        cv = ClaimVerdict(**{k: v for k, v in c.items() if k in FIELDS})
        judged = c.get("judge_verdict") or (
            "SUPPORTED" if c["verdict"] == "UNSUPPORTED" and c["quote"] and c["citation"]
            else c["verdict"])
        cv.verdict, cv.citation_corrected = judged, False
        CitationVerifier._repoint_citation(cv, context)
        cv.valid_quote = CitationVerifier._quote_in_source(cv, context)
        cv.numeric_ok = CitationVerifier._numeric_in_source(cv, context)
        if cv.verdict in ("SUPPORTED", "PARTIAL") and (
                cv.citation is None or not cv.valid_quote or not cv.numeric_ok):
            cv.verdict = "UNSUPPORTED"
        if cv.verdict == "CONTRADICTED" and cv.citation is None:
            cv.verdict = "UNSUPPORTED"
        out.append(cv.to_dict())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--config", default="config/local_16gb_v5.yaml")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = AppConfig.load(args.config)
    with open(run / "questions.csv", newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    answers = _answers(run)
    need = {e["chunk_id"] for a in answers.values() for e in a["extra"]["verification"]["evidence"]}
    texts = chunk_texts(Path(cfg.index.path), need)

    labels = {k: Counter() for k in ("stored", "rescored", "rescored_pruned")}
    flips, changed = Counter(), {}
    for qid, a in answers.items():
        old = a["extra"]["verification"]["claims"]
        new = rescore(old, _context(a, texts))
        for o, n in zip(old, new, strict=True):
            if o["verdict"] != n["verdict"]:
                flips[(o["verdict"], n["verdict"])] += 1
        labels["stored"][label(rows[qid], a["text"], old, False)] += 1
        lab = label(rows[qid], a["text"], new, False)
        labels["rescored"][lab] += 1
        labels["rescored_pruned"][label(rows[qid], a["text"], new, True)] += 1
        if lab != rows[qid]["error_source"]:
            changed[qid] = {"before": rows[qid]["error_source"], "after": lab}
    (run / "quote_rescore.json").write_text(json.dumps(
        {"flips": {f"{a} -> {b}": n for (a, b), n in flips.items()}, "changed": changed},
        indent=1))
    print("claim verdict changes:", {f"{a} -> {b}": n for (a, b), n in flips.items()})
    # questions abstained in the run keep their label
    for qid, r in rows.items():
        if qid.startswith("finder_") and qid not in answers:
            for counts in labels.values():
                counts[r["error_source"]] += 1
    print(f"{'label':40}" + "".join(f"{k:>16}" for k in labels))
    for lab in sorted(set().union(*labels.values()), key=lambda x: -labels["stored"][x]):
        print(f"{lab:40}" + "".join(f"{labels[k][lab]:16}" for k in labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
