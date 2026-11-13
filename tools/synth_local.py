#!/usr/bin/env python3
"""In-session variant of make_synth_qa.py: Claude (the coding agent) writes the Q/A.

    python tools/synth_local.py export      # sample 500 -> out/synth_qa/batches/bNN.md
    # Claude reads each bNN.md and writes out/synth_qa/answers/bNN.json:
    #   {"s0001": {"usable", "reason_if_unusable", "question", "terse", "answer",
    #              "evidence", "answer_type"}, ...}
    # plus out/synth_qa/answers/finder_q.json: {"finder_0001": "<question>", ...}
    python tools/synth_local.py merge       # verify evidence verbatim, write outputs

Same sampling, schema and verification as make_synth_qa.py; no API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_synth_qa as m  # noqa: E402

BATCH = 25
GENERATOR = "claude-opus-5-5 (in-session agent)"


def export() -> None:
    m.OUT.mkdir(parents=True, exist_ok=True)
    path = m.OUT / "sample.json"
    if not path.exists():
        path.write_text(json.dumps(m.sample(500, 7), indent=1))
    picks = json.loads(path.read_text())
    bdir = m.OUT / "batches"
    bdir.mkdir(exist_ok=True)
    for b in range(0, len(picks), BATCH):
        lines = []
        for p in picks[b:b + BATCH]:
            lines.append(f"### {p['sid']} | {p['ticker']} | {p['filing_type']} "
                         f"{p['filing_date']} | item {p['item_number']} | "
                         f"{p['section_title']}\n{p['text']}\n")
        (bdir / f"b{b // BATCH:02d}.md").write_text("\n".join(lines))
    tick = {p["ticker"] for p in picks}
    dev = set(json.loads(Path("eval/finder_split.json").read_text())["dev"])
    notes = [f"{c['id']} | {c['ticker']} | {c['question']}"
             for c in json.loads(Path("eval/finder_semantic.json").read_text())
             if c["ticker"] in tick and c["id"] in dev]
    (bdir / "finder_notes.txt").write_text("\n".join(notes))
    print(f"{len(picks)} passages, {len({p['ticker'] for p in picks})} tickers, "
          f"{-(-len(picks) // BATCH)} batches, {len(notes)} FinDER notes")
    print("style examples:\n" + m.finder_style())


def merge() -> None:
    picks = json.loads((m.OUT / "sample.json").read_text())
    results: dict[str, dict] = {}
    adir = m.OUT / "answers"
    for f in sorted(adir.glob("b*.json")):
        results.update(json.loads(f.read_text()))
    fq = adir / "finder_q.json"
    if fq.exists():
        results.update({f"fq_{k}": {"question": v} for k, v in json.loads(fq.read_text()).items()})
    m.MODEL = GENERATOR
    (m.OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"answers for {sum(not k.startswith('fq_') for k in results)}/{len(picks)} passages")
    m.write_outputs(picks, results)


if __name__ == "__main__":
    {"export": export, "merge": merge}[sys.argv[1]]()
