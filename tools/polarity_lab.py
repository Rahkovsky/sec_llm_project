#!/usr/bin/env python3
"""Q4 lab: can a mechanical check catch flipped polarity that the judge passes?

The judge returns a verbatim quote for every SUPPORTED claim, and the pipeline checks the
quote's figures against the claim (``numeric_ok``). Nothing compares their *polarity*: the
judge marks "competition is expected to decrease" SUPPORTED with a quote saying it is
"expected to continue to increase" (tools/judge_mutants.py, EXPERIMENTS §20.9). This flags a
SUPPORTED claim when, against its quote,

  direction  the claim uses a direction word whose opposite is in the quote and which the
             quote itself lacks ("decrease" vs a quote with "increase" and no "decrease");
             ``verifier.direction_flip``, the check ``verification.direction_check`` runs
  negation   the claim and the quote differ in the number of negations ("not", "no",
             "never", "n't", "without"), counted within the words the two share

and reports the flag rate on planted errors (recall) and on the v6 production claims (false
flags; nearly all are correct, §20.6).

    python tools/polarity_lab.py out/q4 qwen3-14b --prod out/provenance_16gb_v6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.generation.prune import owner_sentence, split_sentences
from sec_llm_project.sec_intel.generation.verifier import direction_flip, is_meta_claim

_NEG = {"not", "no", "never", "without", "neither", "nor", "none", "cannot"}
_W = re.compile(r"[a-z]+(?:n't)?")


def words(t: str) -> list[str]:
    return _W.findall(t.lower().replace("\u2019", "'"))


def negations(ws: list[str]) -> int:
    return sum(w in _NEG or w.endswith("n't") for w in ws)


def flag(claim: str, quote: str) -> str | None:
    d = direction_flip(claim, quote)  # the check the verifier runs (direction_check)
    if d:
        return f"direction:{d}"
    cw, qw = words(claim), words(quote)
    qs = set(qw)
    # compare negations only when the quote covers the claim's content (most words shared);
    # a quote of one clause says nothing about the rest of a long claim
    content = [w for w in cw if len(w) > 3 and w not in _NEG]
    if content and sum(w in qs for w in content) / len(content) >= 0.6 \
            and negations(cw) != negations(qw):
        return f"negation:{negations(cw)}/{negations(qw)}"
    return None


def claims_of(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    return d["claims"] if "claims" in d else d["extra"]["verification"]["claims"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("judge")
    ap.add_argument("--prod", default="out/provenance_16gb_v6")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    root = Path(args.dir)

    muts = json.loads((root / "mutants" / "mutations.json").read_text())
    caught, judge_pass = Counter(), Counter()
    n = Counter()
    for m in muts:
        a = json.loads((root / "originals" / m["id"] / "answer.json").read_text())
        text = a["text"].replace(m["before"], m["after"], 1)
        sents = split_sentences(text)
        p = root / f"{args.judge}_mut" / f"{m['id']}.json"
        if not p.exists():
            continue
        cl = [c for c in claims_of(p) if owner_sentence(c["claim"], sents) == m["index"]
              and not (c.get("meta") or is_meta_claim(c["claim"]))]
        n[m["kind"]] += 1
        passed = bool(cl) and all(c["verdict"] == "SUPPORTED" for c in cl)
        if passed:
            judge_pass[m["kind"]] += 1
            if any(flag(c["claim"], c.get("quote") or "") for c in cl):
                caught[m["kind"]] += 1
    print(f"planted errors the {args.judge} pipeline passed, now flagged:")
    for k in sorted(n):
        print(f"  {k:9} {caught[k]}/{judge_pass[k]} (of {n[k]} planted; "
              f"pipeline + flag catch {n[k] - judge_pass[k] + caught[k]}/{n[k]})")

    flagged, total = [], 0
    for f in sorted(Path(args.prod).glob("finder_*/answer.json")):
        a = json.loads(f.read_text())
        if a["abstained"]:
            continue
        for c in a["extra"].get("verification", {}).get("claims", []):
            if c["verdict"] != "SUPPORTED" or c.get("meta") or not c.get("quote"):
                continue
            total += 1
            r = flag(c["claim"], c["quote"])
            if r:
                flagged.append((f.parent.name, r, c["claim"], c["quote"]))
    print(f"\nv6 SUPPORTED claims flagged: {len(flagged)}/{total} = {len(flagged) / total:.1%} "
          f"{dict(Counter(r.split(':')[0] for _, r, _, _ in flagged))}")
    for q, r, c, qu in flagged[: args.show]:
        print(f"\n{q} {r}\n  claim: {c[:240]}\n  quote: {qu[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
