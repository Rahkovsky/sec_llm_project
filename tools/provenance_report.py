#!/usr/bin/env python3
"""Seminar report over a tools/provenance_dag.py output dir.

Aggregates the per-question DAGs into (a) RAGAS-style corpus metrics computed
with the LOCAL stack, (b) a per-question error-source attribution (retrieval vs
generation vs abstention), and (c) a self-contained out/<dir>/index.html
dashboard linking every question's interactive DAG. Model-free except the
optional answer-relevancy column (local Nomic embeddings, --relevancy).

    python tools/provenance_report.py [--out out/provenance] [--relevancy]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.evaluation.metrics import CONTAINMENT_THRESHOLD, passage_hit
from sec_llm_project.sec_intel.generation.verifier import is_meta_claim

# Modal + forward-looking/opinion hedges: dropping one turns a risk or
# forward-looking disclosure into a bare factual claim. A deliberate LEXICAL
# PROXY (Phase 3 replaces it with a judged check) — word-boundary matched so
# "if" != "difficult" and "certain" (matches "uncertain") is excluded, and the
# polysemous estimate/approximately family is left to the numeric_ok invariant.
HEDGES = ["may", "could", "might", "would", "believe", "believes", "believed",
          "expect", "expects", "expected", "anticipate", "anticipates", "anticipated",
          "intend", "intends", "intended", "potential", "potentially", "likely",
          "subject to"]
_HEDGE_RE = re.compile(r"\b(" + "|".join(HEDGES) + r")\b")
WEIGHTS = {"SUPPORTED": 0.0, "PARTIAL": 1.0, "UNSUPPORTED": 2.0, "CONTRADICTED": 3.0}


def hedges_in(text: str) -> set[str]:
    return set(_HEDGE_RE.findall(text.lower()))


def hardened_hedges(claim: str, quote: str) -> list[str]:
    """Evidence hedges the claim drops entirely — flagged only when the claim
    keeps NO hedge of its own (i.e. states as fact what the filing qualified)."""
    ev = hedges_in(quote)
    return sorted(ev) if ev and not hedges_in(claim) else []


def chunk_texts(index_dir: Path, needed: set[str]) -> dict[str, str]:
    """Full chunk texts for evidence ids (answer.json quotes are truncated).

    Must read the index the run used: a missing catalog silently falls back to
    the truncated quotes, which badly understates context recall."""
    texts: dict[str, str] = {}
    catalog = index_dir / "chunks.jsonl"
    if not catalog.exists():
        print(f"WARNING: {catalog} not found; context recall uses truncated quotes "
              "and is understated", file=sys.stderr)
        return texts
    with open(catalog, encoding="utf-8") as fh:
        for line in fh:
            if len(texts) == len(needed):
                break
            cid = line[line.find('"chunk_id"') + 13:]
            cid = cid[: cid.find('"')]
            if cid in needed:
                texts[cid] = json.loads(line).get("text", "")
    if len(texts) < len(needed):
        print(f"WARNING: {len(needed) - len(texts)}/{len(needed)} evidence chunks not in "
              f"{catalog}", file=sys.stderr)
    return texts


def evidence_ids(a: dict) -> list[str]:
    """Chunks the answer was grounded on, for gold-recall scoring.

    Verification evidence when present; otherwise the full ranked retrieval list
    (recorded since Nov 2026); otherwise the citations (older abstentions carry
    only the first one or two). Full texts are looked up by id, never quotes."""
    ver = a["extra"].get("verification", {}).get("evidence", [])
    if ver:
        return [ev["chunk_id"] for ev in ver]
    if a["extra"].get("retrieval"):
        return [r["chunk_id"] for r in a["extra"]["retrieval"]]
    return [c["chunk_id"] for c in a["citations"]]


def classify(row: dict) -> str:
    """Attribute the dominant error source for one question."""
    if not row.get("answerable", True):  # placebo / negative control
        # an answer whose only claims say the excerpts lack the information is a refusal
        return ("unanswerable: correctly abstained" if row["abstained"] or row["meta_only"]
                else "unanswerable: CONFABULATED")
    if row["abstained"]:
        return ("retrieval miss -> abstained" if row["ev_recall"] is not None
                and row["ev_recall"] < 0.5 else "abstained (gate/judge)")
    if row["ev_recall"] is not None and row["ev_recall"] < 0.5 and row["unsupported"]:
        return "retrieval miss -> ungrounded claims"
    if row["wrong_period"]:
        return "wrong-period evidence"
    if row["contradicted"]:
        return "generation: contradicted claim"
    if row["unsupported"]:
        return "generation: unsupported claims"
    if row["hardened"]:
        return "generation: dropped qualifiers"
    if row["partial"]:
        return "partial support"
    return "clean"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/provenance")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("--config", default="config/local.yaml",
                    help="config the run used: its index supplies full chunk texts, "
                         "its embedder the relevancy column")
    ap.add_argument("--relevance", default="cover", choices=["cover", "strict"],
                    help="gold-relevance rule for error attribution (strict = legacy runs)")
    ap.add_argument("--relevancy", action="store_true",
                    help="add RAGAS answer-relevancy via the --config embedder")
    args = ap.parse_args()

    cases = {c["id"]: c for c in json.loads(Path(args.cases).read_text())}
    answers = {p.parent.name: json.loads(p.read_text())
               for p in sorted(Path(args.out).glob("*/answer.json"))}
    if not answers:
        print("no answers found")
        return 1

    need = {cid for a in answers.values() for cid in evidence_ids(a)}
    from sec_llm_project.sec_intel.core.config import AppConfig
    cfg = AppConfig.from_file(args.config)
    texts = chunk_texts(Path(cfg.index.path), need)

    relev = {}
    if args.relevancy:
        from sec_llm_project.sec_intel.embeddings.factory import build_embedder
        emb = build_embedder(cfg.embedding)
        for qid, a in answers.items():
            if not a["abstained"]:
                q, t = emb.embed_query(a["question"]), emb.embed_query(a["text"])
                relev[qid] = sum(x * y for x, y in zip(q, t, strict=True))

    rows, drops = [], []
    for qid, a in answers.items():
        ver = a["extra"].get("verification", {})
        # Meta claims ("the excerpts do not say X") describe missing evidence, not
        # the filing; the verifier excludes them from scoring, and so does this.
        all_claims = ver.get("claims", [])
        # Re-applied with the current rule, so older runs are scored the same way.
        claims = [c for c in all_claims if not (c.get("meta") or is_meta_claim(c["claim"]))]
        gold = cases.get(qid, {}).get("relevant_passages", [])
        quotes = {ev["chunk_id"]: ev["quote"] for ev in ver.get("evidence", [])}
        quotes.update({c["chunk_id"]: c["quote"] for c in a["citations"]})
        ev_texts = [texts.get(cid, quotes.get(cid, "")) for cid in evidence_ids(a)]

        def recall(cont: float | None, ev_texts=ev_texts, gold=gold) -> float | None:
            if not gold:
                return None
            return sum(any(passage_hit(t, [p], 0.5, cont) for t in ev_texts)
                       for p in gold) / len(gold)
        ev_recall_strict, ev_recall_cover = recall(None), recall(CONTAINMENT_THRESHOLD)
        ev_recall = ev_recall_cover if args.relevance == "cover" else ev_recall_strict

        # Wrong-period proxy: a year named in the question vs cited filing dates
        # (±1y: the 10-K for fiscal t is filed early t+1). 19xx matters: asking
        # about a pre-coverage year is exactly how period-blindness shows up.
        year_m = re.search(r"\b((?:19|20)\d{2})\b", a["question"])
        wrong_period = sum(
            abs(int(c["filing_date"][:4]) - int(year_m.group(1))) > 1
            for c in a["citations"] if c.get("filing_date", "")[:4].isdigit()
        ) if year_m else 0

        row = {"id": qid, "ticker": cases.get(qid, {}).get("ticker", ""),
               "question": a["question"], "abstained": a["abstained"],
               "answerable": cases.get(qid, {}).get("answerable", True),
               "wrong_period": wrong_period, "year_named": bool(year_m),
               "claims": len(claims),
               "meta_only": bool(all_claims) and not claims and not a["abstained"],
               "supported": sum(c["verdict"] == "SUPPORTED" for c in claims),
               "partial": sum(c["verdict"] == "PARTIAL" for c in claims),
               "unsupported": sum(c["verdict"] == "UNSUPPORTED" for c in claims),
               "contradicted": sum(c["verdict"] == "CONTRADICTED" for c in claims),
               "weight": sum(WEIGHTS.get(c["verdict"], 0) for c in claims),
               "ev_recall": ev_recall,
               "ev_recall_strict": ev_recall_strict, "ev_recall_cover": ev_recall_cover, "relevancy": relev.get(qid)}
        row["hardened"] = 0
        qual = a["extra"].get("qualifiers")
        row["qual_judged"] = len(qual["dropped"]) if qual else None
        row["qual_repaired"] = bool(qual and qual.get("repaired"))
        for c in claims:
            if c["verdict"] == "SUPPORTED" and c["quote"]:
                d = hardened_hedges(c["claim"], c["quote"])
                if d:
                    row["hardened"] += 1
                    drops.append({"id": qid, "dropped": " ".join(d),
                                  "claim": c["claim"], "evidence": c["quote"]})
        row["error_source"] = classify(row)
        rows.append(row)

    # ------------------------------------------------------------- aggregates
    n_claims = sum(r["claims"] for r in rows)
    answered = [r for r in rows if not r["abstained"]]
    faith = sum(r["supported"] for r in rows) / n_claims if n_claims else None
    with_gold = [r for r in rows if r["ev_recall"] is not None]
    ctx = {k: (sum(r[k] for r in with_gold) / len(with_gold) if with_gold else None)
           for k in ("ev_recall_strict", "ev_recall_cover")}
    wcer = sum(r["weight"] for r in rows) / n_claims if n_claims else None
    mean_rel = (sum(relev.values()) / len(relev)) if relev else None

    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"  # noqa: E731
    print(f"questions: {len(rows)}  answered: {len(answered)}  "
          f"abstained: {len(rows) - len(answered)}")
    print("RAGAS-style (local stack):")
    print(f"  faithfulness (claim support rate):   {fmt(faith)}")
    print(f"  context recall, strict (legacy):     {fmt(ctx['ev_recall_strict'])}")
    print(f"  context recall, cover:               {fmt(ctx['ev_recall_cover'])}")
    print(f"  answer relevancy (nomic cosine):     {fmt(mean_rel)}")
    print("DAG-level:")
    print(f"  weighted claim error rate:           {fmt(wcer)}")
    print(f"  hardened-qualifier claims (lexical): {sum(r['hardened'] for r in rows)}"
          "  (proxy; 14/73 real when hand-labelled on v3)")
    judged = [r for r in rows if r["qual_judged"] is not None]
    if judged:
        print(f"  dropped qualifiers (judge-checked):  {sum(r['qual_judged'] for r in judged)} "
              f"in {sum(bool(r['qual_judged']) for r in judged)} answers; "
              f"repaired {sum(r['qual_repaired'] for r in judged)} answers")
    unans = [r for r in rows if not r["answerable"]]
    if unans:
        ok = sum(r["abstained"] or r["meta_only"] for r in unans)
        print(f"  correct abstention (placebos):       {ok}/{len(unans)}")
    meta_only = [r["id"] for r in rows if r["meta_only"]]
    print(f"  answered, but only meta claims:     {len(meta_only)} {meta_only[:8]}")
    yearq = [r for r in rows if r["year_named"] and not r["abstained"]]
    if yearq:
        bad = sum(bool(r["wrong_period"]) for r in yearq)
        print(f"  wrong-period answers (year-named):   {bad}/{len(yearq)}")
    print("error sources:")
    for src in sorted({r["error_source"] for r in rows}):
        print(f"  {src:38s} {sum(r['error_source'] == src for r in rows)}")

    out = Path(args.out)
    with open(out / "questions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if drops:
        with open(out / "qualifier_drops.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "dropped", "claim", "evidence"])
            w.writeheader()
            w.writerows(drops)
    (out / "index.html").write_text(_dashboard(rows, faith, ctx["ev_recall_cover"], mean_rel, wcer))
    (out / "error_sources.svg").write_text(_bar_svg(rows))
    print(f"\nwrote {out}/index.html, error_sources.svg, questions.csv"
          + (f", qualifier_drops.csv ({len(drops)} hardened)" if drops else ""))
    return 0


_BAR_COLOR = {"clean": "#16a34a", "partial support": "#d97706"}


def _bar_svg(rows: list[dict], width: int = 640) -> str:
    """Slide-ready horizontal bar chart of error sources (pure SVG, no deps)."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["error_source"]] = counts.get(r["error_source"], 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    bh, gap, left, top = 30, 12, 300, 46
    h = top + len(items) * (bh + gap) + 16
    scale = (width - left - 60) / max(c for _, c in counts.items())
    bars = []
    for i, (src, c) in enumerate(items):
        y = top + i * (bh + gap)
        color = _BAR_COLOR.get(src, "#dc2626" if "retrieval" in src else "#f59e0b"
                               if "generation" in src else "#64748b")
        bars.append(
            f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" class="l">{src}</text>'
            f'<rect x="{left}" y="{y}" width="{max(c * scale, 3):.0f}" height="{bh}" '
            f'rx="4" fill="{color}" opacity="0.85"/>'
            f'<text x="{left + max(c * scale, 3) + 8:.0f}" y="{y + 20}" class="n">{c}</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}"
 viewBox="0 0 {width} {h}"><style>
 text{{font-family:Helvetica,Arial,sans-serif}} .t{{font-size:16px;font-weight:600}}
 .s{{font-size:11px;fill:#475569}} .l{{font-size:12px;fill:#0f172a}}
 .n{{font-size:12px;font-weight:600;fill:#0f172a}}</style>
<text x="14" y="22" class="t">Where answers fail: error sources ({len(rows)} questions)</text>
<text x="14" y="38" class="s">red = retrieval, amber = generation, gray = gating; from claim-provenance DAGs</text>
{"".join(bars)}</svg>"""


_CHIP = {"clean": "#dcfce7", "partial support": "#fef3c7"}


def _dashboard(rows, faith, ctx_recall, mean_rel, wcer) -> str:
    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"  # noqa: E731
    cards = "".join(
        f'<div class="card"><div class="v">{fmt(v)}</div><div class="k">{k}</div></div>'
        for k, v in [("faithfulness<br>(claim support)", faith),
                     ("context recall, cover<br>(vs gold passages)", ctx_recall),
                     ("answer relevancy<br>(nomic cosine)", mean_rel),
                     ("weighted claim<br>error rate", wcer)])
    trs = []
    for r in sorted(rows, key=lambda r: r["id"]):
        chip = _CHIP.get(r["error_source"], "#fee2e2")
        verdicts = (f'✓{r["supported"]} ◐{r["partial"]} ✗{r["unsupported"]}'
                    + (f' ‼{r["contradicted"]}' if r["contradicted"] else ""))
        trs.append(
            f'<tr><td><a href="{r["id"]}/dag.html">{r["id"]}</a></td>'
            f'<td>{r["ticker"]}</td><td class="q">{r["question"][:90]}</td>'
            f'<td>{verdicts if r["claims"] else "—"}</td>'
            f'<td>{fmt(r["ev_recall"])}</td>'
            f'<td><span class="chip" style="background:{chip}">{r["error_source"]}</span></td></tr>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Claim-provenance report</title>
<style>
 body{{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;margin:0;color:#0f172a}}
 header{{padding:16px 24px;border-bottom:1px solid #e2e8f0}} h2{{margin:0 0 2px;font-size:18px}}
 .sub{{color:#475569;font-size:13px}}
 .cards{{display:flex;gap:14px;padding:16px 24px}}
 .card{{border:1px solid #e2e8f0;border-radius:10px;padding:12px 18px;min-width:130px;text-align:center}}
 .card .v{{font-size:24px;font-weight:600}} .card .k{{font-size:11.5px;color:#475569;margin-top:4px}}
 table{{border-collapse:collapse;margin:8px 24px 32px;font-size:13px}}
 th,td{{padding:6px 10px;border-bottom:1px solid #eef2f7;text-align:left;vertical-align:top}}
 th{{color:#475569;font-weight:600}} .q{{max-width:460px;color:#334155}}
 .chip{{padding:2px 9px;border-radius:10px;font-size:12px;white-space:nowrap}}
 a{{color:#4f46e5;text-decoration:none}}
</style></head><body>
<header><h2>Claim-provenance report — EDGAR RAG</h2>
<div class="sub">Aggregate metrics say <b>what</b>; each linked DAG shows <b>why</b> —
evidence → claims → conclusion, with the error source attributed per question.</div></header>
<div class="cards">{cards}</div>
<table><tr><th>question</th><th>ticker</th><th></th><th>claims ✓/◐/✗</th>
<th>ev. recall</th><th>error source</th></tr>{"".join(trs)}</table>
</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
