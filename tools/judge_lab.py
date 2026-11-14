#!/usr/bin/env python3
"""S1 (docs/NEXT_STEPS.md): re-judge stored answers with another judge format.

Takes the answers of an existing run, rebuilds the exact evidence the judge saw
(chunk ids stored in answer.json, full texts from the index), and verifies each
answer again with ``verification.judge_format`` set to FORMAT. Records every judge
call's time and token counts.

    python tools/judge_lab.py run out/provenance_16gb_v5 --config config/local_16gb_v5.yaml \
        --format sentences --out out/judge_lab/sentences
    python tools/judge_lab.py compare out/provenance_16gb_v5 out/judge_lab/sentences

``compare`` re-applies the current quote and figure checks to both judges' claims
(tools/quote_rescore.py), maps both judges' verdicts to answer sentences (the claims format is
mapped by word overlap, as in generation/prune.py) and reports agreement, judge time
and output tokens, and the clean-answer count each judge would give, with and
without pruning.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from provenance_report import chunk_texts, classify, hardened_hedges

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.core.types import (
    Answer,
    Chunk,
    ClaimVerdict,
    FilingMetadata,
    RetrievalResult,
)
from sec_llm_project.sec_intel.generation.prune import (
    owner_sentence,
    prune_unsupported,
    split_sentences,
)
from sec_llm_project.sec_intel.generation.verifier import build_verifier, is_meta_claim
from sec_llm_project.sec_intel.llm import providers
from sec_llm_project.utils.env_config import load_environment

FIELDS = set(ClaimVerdict.__dataclass_fields__)
FAIL = {"UNSUPPORTED", "CONTRADICTED"}


def _answers(run: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(run.glob("finder_*/answer.json")):
        a = json.loads(p.read_text())
        if not a["abstained"]:
            out[p.parent.name] = a
    return out


def _context(a: dict, texts: dict[str, str]) -> list[RetrievalResult]:
    ctx = []
    for i, e in enumerate(a["extra"]["verification"]["evidence"]):
        meta = FilingMetadata(ticker=e["ticker"], company=e["company"],
                              filing_type=e["filing_type"], filing_date=e["filing_date"])
        chunk = Chunk(chunk_id=e["chunk_id"], text=texts.get(e["chunk_id"], e["quote"]),
                      metadata=meta, section_title=e["section_title"],
                      item_number=e["item_number"])
        ctx.append(RetrievalResult(chunk=chunk, score=float(e["score"]), rank=i))
    return ctx


def cmd_run(args) -> int:
    load_environment()
    cfg = AppConfig.load(args.config)
    cfg.verification.judge_format = args.format
    verifier = build_verifier(cfg.verification, cfg.llm)
    answers = _answers(Path(args.run))
    if args.n:
        answers = dict(list(answers.items())[: args.n])
    need = {e["chunk_id"] for a in answers.values() for e in a["extra"]["verification"]["evidence"]}
    texts = chunk_texts(Path(cfg.index.path), need)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, (qid, a) in enumerate(answers.items(), 1):
        dest = out / f"{qid}.json"
        if dest.exists():
            continue
        providers.reset_call_log()
        t0 = time.perf_counter()
        report = verifier.verify(Answer(question=a["question"], text=a["text"]),
                                 _context(a, texts))
        dest.write_text(json.dumps({
            "wall_s": round(time.perf_counter() - t0, 2), "calls": providers.call_log(),
            "claims": [c.to_dict() for c in report.claims]}, indent=1))
        print(f"[{k}/{len(answers)}] {qid} {len(report.claims)} sentences "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    return 0


# ----------------------------------------------------------------- compare
def sentence_verdicts(text: str, claims: list[dict]) -> list[str]:
    """Per answer sentence: S (all scored claims supported), F (all fail), P (mixed or
    partial), M (only meta claims), N (no claim mapped)."""
    sents = split_sentences(text)
    by = [[] for _ in sents]
    for c in claims:
        meta = c.get("meta") or is_meta_claim(c["claim"])
        by[owner_sentence(c["claim"], sents)].append("M" if meta else c["verdict"])
    out = []
    for vs in by:
        scored = [v for v in vs if v != "M"]
        if not vs:
            out.append("N")
        elif not scored:
            out.append("M")
        elif all(v == "SUPPORTED" for v in scored):
            out.append("S")
        elif all(v in FAIL for v in scored):
            out.append("F")
        else:
            out.append("P")
    return out


def label(row: dict, text: str, claims: list[dict], prune: bool) -> str:
    cvs = [ClaimVerdict(**{k: v for k, v in c.items() if k in FIELDS}) for c in claims]
    for c in cvs:
        c.meta = c.meta or is_meta_claim(c.claim)
    if prune:
        cvs = prune_unsupported(text, cvs).kept
    scored = [c for c in cvs if not c.meta]
    ev = row["ev_recall_cover"]
    r = {"answerable": True, "ev_recall": float(ev) if ev not in ("", None) else None,
         "wrong_period": int(row["wrong_period"]), "meta_only": False,
         "abstained": not scored,
         "unsupported": sum(c.verdict == "UNSUPPORTED" for c in scored),
         "contradicted": sum(c.verdict == "CONTRADICTED" for c in scored),
         "partial": sum(c.verdict == "PARTIAL" for c in scored),
         "hardened": sum(bool(hardened_hedges(c.claim, c.quote))
                         for c in scored if c.verdict == "SUPPORTED" and c.quote)}
    return classify(r)


def cmd_compare(args) -> int:
    from quote_rescore import rescore  # the current quote/figure checks, for both judges

    run, new = Path(args.run), Path(args.new)
    with open(run / "questions.csv", newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    answers = _answers(run)
    cfg = AppConfig.load(args.config)
    texts = chunk_texts(Path(cfg.index.path), {
        e["chunk_id"] for a in answers.values() for e in a["extra"]["verification"]["evidence"]})
    pairs, t_old, t_new, tok_old, tok_new = Counter(), [], [], [], []
    labels = {k: Counter() for k in ("old", "new", "old_pruned", "new_pruned")}
    n = 0
    for qid, a in answers.items():
        p = new / f"{qid}.json"
        if not p.exists():
            continue
        n += 1
        nd = json.loads(p.read_text())
        ctx = _context(a, texts)
        old_claims = rescore(a["extra"]["verification"]["claims"], ctx)
        nd["claims"] = rescore(nd["claims"], ctx)
        for o, w in zip(sentence_verdicts(a["text"], old_claims),
                        sentence_verdicts(a["text"], nd["claims"]), strict=True):
            pairs[(o, w)] += 1
        tj = json.loads((run / qid / "timings.json").read_text())["calls"]
        t_old.append(sum(c["wall_s"] for c in tj if c["stage"] == "judge"))
        tok_old.append(sum(c["out_tokens"] or 0 for c in tj if c["stage"] == "judge"))
        t_new.append(sum(c["wall_s"] for c in nd["calls"] if c["stage"] == "judge"))
        tok_new.append(sum(c["out_tokens"] or 0 for c in nd["calls"] if c["stage"] == "judge"))
        for key, claims in (("old", old_claims), ("new", nd["claims"])):
            labels[key][label(rows[qid], a["text"], claims, False)] += 1
            labels[key + "_pruned"][label(rows[qid], a["text"], claims, True)] += 1

    print(f"answers compared: {n}")
    print(f"judge time per answer:   old {st.mean(t_old):.1f}s  new {st.mean(t_new):.1f}s")
    print(f"judge output tokens:     old {st.mean(tok_old):.0f}  new {st.mean(tok_new):.0f}")
    keys = "SPFMN"
    print("sentence verdicts, old (rows) x new (cols); S supported, P mixed, F failed, "
          "M meta only, N no claim")
    print("      " + "".join(f"{k:>6}" for k in keys))
    for o in keys:
        print(f"{o:>6}" + "".join(f"{pairs[(o, w)]:6}" for w in keys))
    scored = {(o, w): v for (o, w), v in pairs.items() if o in "SPF" and w in "SPF"}
    agree = sum(v for (o, w), v in scored.items() if (o == "S") == (w == "S"))
    print(f"pass/fail agreement on scored sentences (S vs P/F): "
          f"{agree}/{sum(scored.values())} = {agree / max(1, sum(scored.values())):.3f}")
    print(f"{'label':40}" + "".join(f"{k:>12}" for k in labels))
    for lab in sorted(set().union(*labels.values()), key=lambda x: -labels["old"][x]):
        print(f"{lab:40}" + "".join(f"{labels[k][lab]:12}" for k in labels))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run")
    r.add_argument("--config", default="config/local_16gb_v5.yaml")
    r.add_argument("--format", default="sentences", choices=["claims", "sentences", "verdicts"])
    r.add_argument("--out", required=True)
    r.add_argument("-n", type=int, default=0, help="first n answers only (0 = all)")
    c = sub.add_parser("compare")
    c.add_argument("run")
    c.add_argument("new")
    c.add_argument("--config", default="config/local_16gb_v5.yaml",
                   help="its index supplies the chunk texts for the quote checks")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
