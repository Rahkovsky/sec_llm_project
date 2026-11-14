#!/usr/bin/env python3
"""Placebo answers: does the answer assert the false premise? (issue #2)

A placebo question cannot be answered from any filing (a 10-K that does not exist,
a company that files none, a bankruptcy that never happened). The report's
built-in counter scores a placebo as refused only when the pipeline abstained or
every sentence is a meta-claim ("the excerpts do not contain ..."), which marks
refusals written in ordinary prose as failures.

This asks the question that matters instead. Each placebo case carries a
``confabulation`` field: what a wrong answer would claim (e.g. "describes
impairment charges as resulting from, or recorded after, a Tesla bankruptcy
filing"). For every placebo answer the judge model is asked whether the answer
makes that claim. An abstained answer asserts nothing and needs no call.

    python tools/placebo_check.py out/provenance_16gb_v3 out/provenance_16gb_v4
    python tools/placebo_check.py --validate out/provenance_16gb_v3 \
        out/provenance_16gb_v3_gold out/provenance_16gb_v4   # vs eval/placebo_labels.json

Writes <run>/placebo_check.json, which tools/provenance_report.py reports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.llm.base import Message
from sec_llm_project.sec_intel.llm.factory import build_llm
from sec_llm_project.utils.env_config import load_environment

PROMPT = """A question was asked that cannot be answered from the company's SEC filings, \
because its premise is false or the information does not exist in them.

QUESTION: {question}

A WRONG answer would do this: it {confabulation}.

ANSWER GIVEN:
{answer}

Does the answer do what a wrong answer would do? Saying that the information is not \
available, or reporting other true facts that do not depend on the false premise, \
does NOT count. Reply with one word, YES or NO, then one sentence of reason."""


def judge(llm, case: dict, answer_text: str) -> tuple[bool | None, str]:
    """True when the judge says the answer asserts the premise; None if unparseable."""
    out = llm.generate([Message("user", PROMPT.format(
        question=case["question"], confabulation=case["confabulation"],
        answer=answer_text))], temperature=0.0, max_tokens=120).text.strip()
    m = re.match(r"\W*(YES|NO)\b", out, re.IGNORECASE)
    return (m.group(1).upper() == "YES" if m else None), out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cases", default="eval/gold_plus_placebo.json")
    ap.add_argument("--config", default="config/local_16gb_v4.yaml")
    ap.add_argument("--validate", action="store_true",
                    help="compare with the hand labels in eval/placebo_labels.json")
    args = ap.parse_args()
    load_environment()
    llm = build_llm(AppConfig.load(args.config).verification.judge)
    placebos = [c for c in json.loads(Path(args.cases).read_text())
                if not c.get("answerable", True)]
    missing = [c["id"] for c in placebos if not c.get("confabulation")]
    if missing:
        raise SystemExit(f"cases without a 'confabulation' field: {missing}")

    labels = {}
    if args.validate:
        for it in json.loads(Path("eval/placebo_labels.json").read_text())["items"]:
            labels[(Path(it["run"]).name, it["id"])] = it["premise_asserted"]
    agree = total = 0
    for run in args.runs:
        rows = {}
        for c in placebos:
            p = Path(run) / c["id"] / "answer.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text())
            if a.get("abstained"):
                rows[c["id"]] = {"abstained": True, "premise_asserted": False, "judge": ""}
            else:
                asserted, raw = judge(llm, c, a["text"])
                rows[c["id"]] = {"abstained": False, "premise_asserted": asserted, "judge": raw}
            want = labels.get((Path(run).name, c["id"]))
            if want is not None:
                total += 1
                agree += rows[c["id"]]["premise_asserted"] == want
        (Path(run) / "placebo_check.json").write_text(json.dumps(rows, indent=1))
        ok = sum(r["premise_asserted"] is False for r in rows.values())
        unclear = sum(r["premise_asserted"] is None for r in rows.values())
        print(f"{run}: premise not asserted in {ok}/{len(rows)} placebo answers"
              + (f" ({unclear} unparseable)" if unclear else ""), flush=True)
        for qid, r in rows.items():
            if not r["abstained"]:
                print(f"   {qid}: asserted={r['premise_asserted']}  {r['judge'][:110]}")
    if args.validate:
        print(f"agreement with hand labels: {agree}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
