#!/usr/bin/env python3
"""Q4/S4 (docs/NEXT_STEPS.md): does the judge catch a planted error, and would a smaller one?

v6 judges almost every claim SUPPORTED, so agreement on its own claims says little about
whether the judge catches wrong ones. This plants one known error in a cited sentence of
each sampled answer and re-judges the originals and the mutants with ``tools/judge_lab.py``.

Mutations (each makes the sentence false against its evidence):
  figure     a number (not a year, not a citation) scaled by 1.3, so the new value is in no
             cited passage
  direction  increased <-> decreased, higher <-> lower, ...
  negation   "is"/"was"/"will"/"may"/... followed by "not"

    python tools/judge_mutants.py make out/provenance_16gb_v6 out/q4/mutants -n 80
    python tools/judge_lab.py run out/q4/mutants --config CFG --format verdicts --out out/q4/<judge>_mut
    python tools/judge_lab.py run out/q4/originals --config CFG --format verdicts --out out/q4/<judge>_orig
    python tools/judge_mutants.py score out/q4 qwen3-14b qwen3-8b gemma4-12b

``score`` reports, per judge, the share of mutated sentences it does not pass (by the raw
judge verdict and after the pipeline's quote and figure checks), the share of the
originals' sentences it does not pass, and its pass/fail agreement with the production run.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sec_llm_project.sec_intel.generation.prune import owner_sentence, split_sentences
from sec_llm_project.sec_intel.generation.verifier import is_meta_claim

_FLIP = {"increased": "decreased", "decreased": "increased", "increase": "decrease",
         "decrease": "increase", "higher": "lower", "lower": "higher", "rose": "fell",
         "fell": "rose", "grew": "declined", "declined": "grew", "more": "less",
         "above": "below", "below": "above", "gain": "loss", "favorable": "unfavorable",
         "unfavorable": "favorable"}
_NEG = re.compile(r"\b(is|are|was|were|will|may|can|could|has|have|does|did)\b(?! not)")
# an amount: "$" before it, or a unit after it (not dates, names like "F5", "10b5-1", "24/7")
_NUM = re.compile(r"(?<![\w\[,./-])(\$?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
                  r"(?=(?:%| ?percent| million| billion| thousand| years?\b| employees))"
                  r"|(?<![\w\[,./-])(\$)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w/-])")
_CITE = re.compile(r"\s*\[[\d,\s]+\]")


def _scale(m: re.Match) -> str | None:
    g = m.groups()
    cur, whole, frac = (g[0], g[1], g[2]) if g[1] is not None else (g[3], g[4], g[5])
    frac = frac or ""
    v = float(whole.replace(",", "") + frac)
    if not frac and 1990 <= v <= 2035:  # a year
        return None
    if v == 0:
        return None
    nv = v * 1.3
    if frac:
        s = f"{nv:.{len(frac) - 1}f}"
    elif "," in whole:
        s = f"{round(nv):,}"
    else:
        s = str(round(nv))
    return None if s == whole + frac else (cur or "") + s  # "$1" * 1.3 rounds to "$1"


def mutate(sentence: str, kind: str, rng: random.Random) -> str | None:
    body = _CITE.sub(lambda m: "\0" * len(m.group()), sentence)  # protect citations
    if kind == "figure":
        ms = [m for m in _NUM.finditer(body) if _scale(m)]
        if not ms:
            return None
        m = rng.choice(ms)
        return sentence[:m.start()] + _scale(m) + sentence[m.end():]
    if kind == "direction":
        ws = [m for m in re.finditer(r"\b[a-z]+\b", body) if m.group() in _FLIP]
        if not ws:
            return None
        m = rng.choice(ws)
        return sentence[:m.start()] + _FLIP[m.group()] + sentence[m.end():]
    ms = list(_NEG.finditer(body))
    if not ms:
        return None
    m = ms[0]
    return sentence[:m.end()] + " not" + sentence[m.end():]


def cmd_make(args) -> int:
    rng = random.Random(args.seed)
    src = Path(args.run)
    pool = []
    for p in sorted(src.glob("finder_*/answer.json")):
        a = json.loads(p.read_text())
        if a["abstained"] or not a["extra"].get("verification", {}).get("evidence"):
            continue
        pool.append((p.parent.name, a))
    rng.shuffle(pool)
    out, orig = Path(args.out), Path(args.out).parent / "originals"
    kinds = ["figure", "direction", "negation"]
    made, log = Counter(), []
    for qid, a in pool:
        if sum(made.values()) >= args.n:
            break
        sents = split_sentences(a["text"])
        cand = [i for i, s in enumerate(sents) if _CITE.search(s) and not is_meta_claim(s)]
        # the kind with the fewest so far that this answer allows
        quota = -(-args.n // len(kinds))  # an equal share of each kind
        for kind in sorted((k for k in kinds if made[k] < quota),
                           key=lambda k: (made[k], rng.random())):
            opts = [(i, m) for i in cand if (m := mutate(sents[i], kind, rng))]
            if opts:
                i, new = rng.choice(opts)
                break
        else:
            continue
        made[kind] += 1
        b = json.loads(json.dumps(a))
        b["text"] = a["text"].replace(sents[i], new, 1)
        for d, ans in ((out, b), (orig, a)):
            (d / qid).mkdir(parents=True, exist_ok=True)
            (d / qid / "answer.json").write_text(json.dumps(ans, indent=1))
            shutil.copy(src / qid / "timings.json", d / qid / "timings.json")
        log.append({"id": qid, "kind": kind, "index": i, "before": sents[i], "after": new})
    for d in (out, orig):
        shutil.copy(src / "questions.csv", d / "questions.csv")
    (out / "mutations.json").write_text(json.dumps(log, indent=1))
    print(f"{len(log)} mutants {dict(made)} -> {out}; originals -> {orig}")
    return 0


def _sentence_pass(text: str, claims: list[dict], field: str) -> list[bool | None]:
    """Per sentence: True if every scored claim mapped to it passes, False if any fails,
    None if no scored claim maps to it."""
    sents = split_sentences(text)
    by: list[list[bool]] = [[] for _ in sents]
    for c in claims:
        if c.get("meta") or is_meta_claim(c["claim"]):
            continue
        v = c.get(field) or c["verdict"]
        by[owner_sentence(c["claim"], sents)].append(v == "SUPPORTED")
    return [all(x) if x else None for x in by]


def cmd_score(args) -> int:
    root = Path(args.dir)
    muts = {m["id"]: m for m in json.loads((root / "mutants" / "mutations.json").read_text())}
    orig_text = {q: json.loads((root / "originals" / q / "answer.json").read_text())
                 for q in muts}
    prod = {q: a["extra"]["verification"]["claims"] for q, a in orig_text.items()}
    print(f"{len(muts)} mutants: {dict(Counter(m['kind'] for m in muts.values()))}\n")
    print("| judge | n | caught, raw judge | caught, after checks | by kind (after checks) "
          "| originals' sentences failed | agree with prod (orig) | judge s/answer |")
    print("|---|---|---|---|---|---|---|---|")
    for judge in args.judges:
        mdir, odir = root / f"{judge}_mut", root / f"{judge}_orig"
        raw = chk = n = 0
        kind = Counter()
        kind_n = Counter()
        secs = []
        for q, m in muts.items():
            p = mdir / f"{q}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            text = orig_text[q]["text"].replace(m["before"], m["after"], 1)
            r = _sentence_pass(text, d["claims"], "judge_verdict")[m["index"]]
            c = _sentence_pass(text, d["claims"], "verdict")[m["index"]]
            n += 1
            raw += r is False
            chk += c is False
            kind_n[m["kind"]] += 1
            kind[m["kind"]] += c is False
            secs.append(d["wall_s"])
        fail = tot = agree = cmp_ = 0
        for q in muts:
            p = odir / f"{q}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            text = orig_text[q]["text"]
            mine = _sentence_pass(text, d["claims"], "verdict")
            ref = _sentence_pass(text, prod[q], "verdict")
            for a, b in zip(mine, ref, strict=True):
                if a is not None:
                    tot += 1
                    fail += a is False
                if a is not None and b is not None:
                    cmp_ += 1
                    agree += a == b
            secs.append(d["wall_s"])
        by = ", ".join(f"{k} {kind[k]}/{kind_n[k]}" for k in sorted(kind_n))
        print(f"| {judge} | {n} | {raw}/{n} = {raw / max(n, 1):.0%} | {chk}/{n} = "
              f"{chk / max(n, 1):.0%} | {by} | {fail}/{tot} | {agree}/{cmp_} | "
              f"{st.median(secs) if secs else 0:.0f} |")
    if args.show:
        j = args.judges[0]
        print(f"\nmissed by {j} (after checks):")
        for q, m in muts.items():
            p = root / f"{j}_mut" / f"{q}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            text = orig_text[q]["text"].replace(m["before"], m["after"], 1)
            if _sentence_pass(text, d["claims"], "verdict")[m["index"]] is not False:
                print(f"  {q} [{m['kind']}] {m['after'][:200]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("run")
    mk.add_argument("out")
    mk.add_argument("-n", type=int, default=80)
    mk.add_argument("--seed", type=int, default=11)
    sc = sub.add_parser("score")
    sc.add_argument("dir")
    sc.add_argument("judges", nargs="+")
    sc.add_argument("--show", action="store_true")
    args = ap.parse_args()
    return cmd_make(args) if args.cmd == "make" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
