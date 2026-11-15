"""S2 lab: could claims copied from the cited text skip the judge? (docs/NEXT_STEPS.md S2)

A claim is "copied" when one sentence of a passage the answer cites (or two adjacent
sentences) contains at least ``--min-cover`` of the claim's content words, every number in
the claim, and the same hedge/negation words (``verifier._SENSITIVE_WORDS``) in both
directions: a claim that drops "may" or adds "not" is never copied. Such a claim would be
marked SUPPORTED without a model call.

The lab reads finished runs (no model calls) and reports, per threshold, how many judged
claims would skip the judge and how the judge actually labelled them. A skipped claim the
judge did not support is a potential false accept; hand labels (``--labels``, from
eval/unsupported_labels_v5.json) say whether it was really unsupported.

    python tools/copy_skip_lab.py out/provenance_16gb_v5 --labels eval/unsupported_labels_v5.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sec_llm_project.sec_intel.generation.verifier import (
    _SENSITIVE_WORDS,
    _keywords,
    _sentences,
)

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]+")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text)}


# direction words: a copied claim that turns "increased" into "decreased" must not skip
_DIRECTION = {"increase", "increased", "increases", "increasing", "decrease", "decreased",
              "decreases", "decreasing", "higher", "lower", "rose", "fell", "grew", "declined",
              "up", "down", "gain", "gains", "loss", "losses", "above", "below", "before",
              "after", "favorable", "unfavorable", "positive", "negative", "excluding",
              "including", "except"}
_GUARD = _SENSITIVE_WORDS | _DIRECTION


def sensitive(text: str) -> Counter[str]:
    """Hedge, negation and direction words with their counts (one dropped "could" of two
    must still differ)."""
    return Counter(w for w in _WORD_RE.findall(text.lower()) if w in _GUARD)


_TOK_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+")


def tokens(text: str) -> list[str]:
    return [t.replace(",", "").rstrip(".") for t in _TOK_RE.findall(text.lower())]


def guards_in_context(claim: str, window: str) -> bool:
    """Every hedge, negation or direction word of the claim sits next to the same word
    (before or after) somewhere in the window, and every figure next to both of its words: "will not be" is not backed by a
    neighbouring "did not recognize", nor "$2811 billion" by "2,811 MWs"."""
    ct, wt = tokens(claim), tokens(window)
    pairs = set(itertools.pairwise(wt))
    for i, t in enumerate(ct):
        if t in _GUARD or t[0].isdigit():
            before = i > 0 and (ct[i - 1], t) in pairs
            after = i + 1 < len(ct) and (t, ct[i + 1]) in pairs
            # a figure needs both neighbours (the unit after it matters: "$2,811 million"
            # vs "2,811 MWs"); a word needs one
            ok = (before or i == 0) and (after or i + 1 == len(ct)) if t[0].isdigit() \
                else (before or after)
            if not ok:
                return False
    return True


def number_seq(text: str) -> list[str]:
    return [n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text)]


def in_order(sub: list[str], seq: list[str]) -> bool:
    """The claim's figures appear in the passage in the same order of first appearance."""
    first = list(dict.fromkeys(x for x in seq if x in set(sub)))
    return list(dict.fromkeys(sub)) == first


def windows(passage: str, kw: set[str]) -> list[str]:
    """Single sentences, and adjacent pairs where each sentence holds at least a quarter of
    the claim's words (so a neighbour's "not" cannot balance a negation the claim added)."""
    s = _sentences(" ".join(passage.split()))
    need = max(2, len(kw) // 4)
    return s + [a + " " + b for a, b in itertools.pairwise(s)
                if len(kw & _keywords(a)) >= need and len(kw & _keywords(b)) >= need]


def copied(claim: str, passages: list[str], min_cover: float) -> bool:
    kw = _keywords(claim)
    if len(kw) < 4:  # too short to tell copying from coincidence
        return False
    nums, sens = numbers(claim), sensitive(claim)
    for p in passages:
        for w in windows(p, kw):
            if len(kw & _keywords(w)) / len(kw) < min_cover:
                continue
            if not nums <= numbers(w) or not in_order(number_seq(claim), number_seq(w)):
                continue  # every figure present, in the passage's order (no swapped figures)
            # the window's hedges must all be in the claim and vice versa; a window
            # with a hedge the claim lacks is the dropped-qualifier case
            if sensitive(w) == sens and guards_in_context(claim, w):
                return True
    return False


def load_chunks(ids: set[str], path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            cid = line[14:line.index('"', 14)] if line.startswith('{"chunk_id": "') else None
            if cid in ids:
                out[cid] = json.loads(line)["text"]
    return out


def main() -> None:  # noqa: PLR0912
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--chunks", default="data/sec_index_v4/chunks.jsonl")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--thresholds", default="1.0,0.95,0.9,0.85")
    ap.add_argument("--show", type=int, default=8, help="examples of skipped non-SUPPORTED")
    args = ap.parse_args()

    answers = []
    for f in sorted(Path(args.run).glob("*/answer.json")):
        a = json.loads(f.read_text())
        ver = (a.get("extra") or {}).get("verification") or {}
        if a.get("abstained") or not ver.get("claims"):
            continue
        cited = [c["chunk_id"] for c in a.get("citations", [])]
        answers.append((f.parent.name, cited, ver["claims"]))
    need = {cid for _, cited, _ in answers for cid in cited}
    texts = load_chunks(need, Path(args.chunks))
    print(f"{len(answers)} answers, {len(need)} cited passages ({len(texts)} found)")

    labels = {}
    if args.labels:
        raw = json.loads(Path(args.labels).read_text())
        items = raw.get("items", raw) if isinstance(raw, dict) else raw
        for it in items:
            labels[(it["id"], it["claim"][:60])] = it.get("label")

    claims = [(qid, c, [texts[x] for x in cited if x in texts])
              for qid, cited, cl in answers for c in cl if not c.get("meta")]
    print(f"{len(claims)} judged non-meta claims; verdicts "
          f"{dict(Counter(c['verdict'] for _, c, _ in claims))}\n")
    print("| min cover | skip judge | of judged | skipped: SUPPORTED | PARTIAL | UNSUPPORTED/"
          "CONTRADICTED | hand label of those |")
    print("|---|---|---|---|---|---|---|")
    for t in [float(x) for x in args.thresholds.split(",")]:
        skipped = [(qid, c) for qid, c, ps in claims if copied(c["claim"], ps, t)]
        v = Counter(c["verdict"] for _, c in skipped)
        bad = [(q, c) for q, c in skipped if c["verdict"] not in ("SUPPORTED", "PARTIAL")]
        hand = Counter(labels.get((q, c["claim"][:60]), "?") for q, c in bad)
        print(f"| {t:.2f} | {len(skipped)} | {len(skipped) / len(claims):.1%} | "
              f"{v['SUPPORTED']} | {v['PARTIAL']} | "
              f"{v['UNSUPPORTED'] + v['CONTRADICTED']} | {dict(hand)} |")
        if t == min(float(x) for x in args.thresholds.split(",")):
            last = [(q, c) for q, c in skipped if c["verdict"] != "SUPPORTED"]
    # answers whose every judged claim is copied: the judge call itself is skipped
    t0 = float(args.thresholds.split(",")[0])
    for t in [float(x) for x in args.thresholds.split(",")]:
        full = sum(all(copied(c["claim"], [texts[x] for x in cited if x in texts], t)
                       for c in cl if not c.get("meta")) for _, cited, cl in answers)
        print(f"min cover {t:.2f}: {full}/{len(answers)} answers need no judge call")
    del t0

    # adversarial: mutate copied SUPPORTED claims; none of the mutants may still be copied
    swaps = [("increased", "decreased"), ("decreased", "increased"), ("higher", "lower"),
             ("lower", "higher"), ("may ", ""), ("could ", ""), (" not ", " "),
             ("is ", "is not "), ("will ", "will not ")]
    tested = accepted = 0
    examples = []
    for qid, c, ps in claims:
        if c["verdict"] != "SUPPORTED" or not copied(c["claim"], ps, min(
                float(x) for x in args.thresholds.split(","))):
            continue
        muts = [c["claim"].replace(a, b, 1) for a, b in swaps if a in c["claim"]]
        seq = number_seq(c["claim"])
        if len(set(seq)) >= 2:  # swap the first two different figures
            a, b = seq[0], next(x for x in seq if x != seq[0])
            muts.append(c["claim"].replace(a, "\0", 1).replace(b, a, 1).replace("\0", b, 1))
        if seq:  # change one digit
            m = re.sub(r"\d", lambda d: str((int(d.group()) + 1) % 10), c["claim"], count=1)
            muts.append(m)
        for m in muts:
            if m == c["claim"]:
                continue
            tested += 1
            if copied(m, ps, min(float(x) for x in args.thresholds.split(","))):
                accepted += 1
                examples.append((qid, m))
    print(f"\nadversarial: {accepted}/{tested} mutated claims still count as copied")
    for qid, m in examples[:5]:
        print(f"  {qid}: {m[:160]}")

    for q, c in last[: args.show]:
        print(f"\n{q} [{c['verdict']}] {c['claim']}\n   quote: {c.get('quote', '')[:200]}")


if __name__ == "__main__":
    main()
