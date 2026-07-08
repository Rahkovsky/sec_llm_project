#!/usr/bin/env python3
"""Emit out/<dir>/showcase.md: 5 pinned exemplar questions in full detail plus
sample-wide aggregates — the seminar handout behind the PDF slides.

    python tools/seminar_showcase.py [--ids finder_0020 ...] [--out out/provenance]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Pinned exemplars: one per failure mode (override with --ids after a full run).
DEFAULT_IDS = ["finder_0020", "finder_0017", "finder_0019", "finder_0002", "finder_0001"]

STORY = {
    "clean": "Everything worked: every claim traces to a filing passage.",
    "generation: unsupported claims":
        "Retrieval found the right filings, but the generator added claims the "
        "evidence does not back — the DAG isolates exactly which ones.",
    "abstained (gate/judge)":
        "Retrieval was fine, but most claims failed verification, so the "
        "groundedness gate withdrew the answer — the verifier as a safety net.",
    "retrieval miss -> ungrounded claims":
        "The gold evidence was never retrieved; the generator built an answer on "
        "tangential passages instead. Claims can look grounded (they cite *some* "
        "filing text) while missing what the question asked — gold recall 0 plus "
        "unsupported claims expose the drift.",
    "retrieval miss -> abstained":
        "The gold evidence was never retrieved and the model itself declared "
        "insufficient evidence — the cheapest correct behavior.",
    "generation: dropped qualifiers":
        "Claims are supported but state as fact what the filing hedged "
        "(may/could/expects) — risk language hardened into assertion.",
    "unanswerable: correctly abstained":
        "Negative control: the question cannot be answered from the corpus "
        "(future/uncovered/false premise) and the system correctly refused.",
    "unanswerable: CONFABULATED":
        "Negative control FAILED: the question cannot be answered from the "
        "corpus, yet the system produced a confident answer — the DAG shows "
        "what it leaned on instead.",
}


def q_section(qid: str, out: Path, case: dict, row: dict) -> str:
    a = json.loads((out / qid / "answer.json").read_text())
    ver = a["extra"].get("verification", {})
    cited = {c["chunk_id"] for c in a["citations"]}
    md = [f"## {qid} — {row['ticker']} · *{row['error_source']}*",
          f"**Question.** {a['question']}",
          f"\n**Category (FinDER).** {case.get('category', '?')} · gold-passage "
          f"recall of retrieval: **{float(row['ev_recall']):.2f}**"
          if row["ev_recall"] else "",
          f"\n**What happened.** {STORY.get(row['error_source'], '')}"]

    md.append("\n**Retrieved evidence** (dashed = never cited by a claim):")
    for ev in ver.get("evidence", []) or a["citations"]:
        mark = "cited" if ev["chunk_id"] in cited else "UNUSED"
        md.append(f"- [{mark}] {ev['ticker']} {ev['filing_type']} {ev['filing_date']} "
                  f"item {ev['item_number']} — “{ev['quote'][:110]}…”")

    if a["abstained"]:
        md.append(f"\n**Outcome: abstained** — “{a['abstain_reason']}”")
    else:
        md.append(f"\n**Answer** (confidence {a['confidence']:.2f}): "
                  f"{a['text'][:400]}{'…' if len(a['text']) > 400 else ''}")

    if ver.get("claims"):
        md.append("\n**Claim-by-claim verdicts** (judge: "
                  f"{ver.get('judge_model', '?')}; quotes verified verbatim):")
        md.append("| verdict | claim | verified quote from filing |")
        md.append("|---|---|---|")
        for c in ver["claims"]:
            quote = c["quote"][:80] + "…" if len(c["quote"]) > 80 else c["quote"] or "—"
            md.append(f"| {c['verdict']} | {c['claim'][:100]} | {quote} |")

    md.append(f"\n**Figure.** `{qid}/dag.pdf` (vector) · interactive: `{qid}/dag.html`\n")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/provenance")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    args = ap.parse_args()
    out = Path(args.out)

    cases = {c["id"]: c for c in json.loads(Path(args.cases).read_text())}
    with open(out / "questions.csv") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}

    n = len(rows)
    abst = sum(r["abstained"] == "True" for r in rows.values())
    claims = sum(int(r["claims"]) for r in rows.values())
    sup = sum(int(r["supported"]) for r in rows.values())
    srcs: dict[str, int] = {}
    for r in rows.values():
        srcs[r["error_source"]] = srcs.get(r["error_source"], 0) + 1

    md = ["# Claim-provenance showcase — EDGAR RAG (FinDER gold set)",
          f"\n## Sample statistics ({n} questions)",
          f"- answered {n - abst}, abstained {abst}",
          f"- claims judged: {claims}; supported {sup} "
          f"(faithfulness / claim-support rate **{sup / claims:.2f}**)" if claims else "",
          "- error sources:"]
    md += [f"  - {k}: **{v}**" for k, v in sorted(srcs.items(), key=lambda kv: -kv[1])]
    md.append("\nAggregate figure: `error_sources.svg` · per-question table: "
              "`questions.csv` · dashboard: `index.html`\n\n---\n")

    for qid in args.ids:
        if qid in rows:
            md.append(q_section(qid, out, cases.get(qid, {}), rows[qid]))
            md.append("---\n")

    (out / "showcase.md").write_text("\n".join(md))
    print(f"wrote {out}/showcase.md ({len(args.ids)} exemplars, {n} questions aggregated)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
