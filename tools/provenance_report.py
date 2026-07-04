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

from sec_llm_project.sec_intel.evaluation.metrics import token_recall

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


def chunk_texts(out_dir: Path, needed: set[str]) -> dict[str, str]:
    """Full chunk texts for evidence ids (answer.json quotes are truncated)."""
    texts: dict[str, str] = {}
    catalog = Path("data/sec_index/chunks.jsonl")
    if not catalog.exists():
        return texts
    with open(catalog, encoding="utf-8") as fh:
        for line in fh:
            if len(texts) == len(needed):
                break
            cid = line[line.find('"chunk_id"') + 13:]
            cid = cid[: cid.find('"')]
            if cid in needed:
                texts[cid] = json.loads(line).get("text", "")
    return texts


def classify(row: dict) -> str:
    """Attribute the dominant error source for one question."""
    if row["abstained"]:
        return ("retrieval miss -> abstained" if row["ev_recall"] is not None
                and row["ev_recall"] < 0.5 else "abstained (gate/judge)")
    if row["ev_recall"] is not None and row["ev_recall"] < 0.5 and row["unsupported"]:
        return "retrieval miss -> ungrounded claims"
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
    ap.add_argument("--relevancy", action="store_true",
                    help="add RAGAS answer-relevancy via local Nomic embeddings")
    args = ap.parse_args()

    cases = {c["id"]: c for c in json.loads(Path(args.cases).read_text())}
    answers = {p.parent.name: json.loads(p.read_text())
               for p in sorted(Path(args.out).glob("finder_*/answer.json"))}
    if not answers:
        print("no answers found")
        return 1

    need = {ev["chunk_id"] for a in answers.values()
            for ev in a["extra"].get("verification", {}).get("evidence", [])}
    texts = chunk_texts(Path(args.out), need)

    relev = {}
    if args.relevancy:
        from sec_llm_project.sec_intel.core.config import AppConfig
        from sec_llm_project.sec_intel.embeddings.factory import build_embedder
        emb = build_embedder(AppConfig.from_file("config/local.yaml").embedding)
        for qid, a in answers.items():
            if not a["abstained"]:
                q, t = emb.embed_query(a["question"]), emb.embed_query(a["text"])
                relev[qid] = sum(x * y for x, y in zip(q, t, strict=True))

    rows, drops = [], []
    for qid, a in answers.items():
        ver = a["extra"].get("verification", {})
        claims = ver.get("claims", [])
        gold = cases.get(qid, {}).get("relevant_passages", [])
        ev_texts = [texts.get(ev["chunk_id"], ev["quote"])
                    for ev in ver.get("evidence", [])] or [c["quote"] for c in a["citations"]]
        ev_recall = (sum(any(token_recall(t, p) >= 0.5 for t in ev_texts) for p in gold)
                     / len(gold)) if gold else None

        row = {"id": qid, "ticker": cases.get(qid, {}).get("ticker", ""),
               "question": a["question"], "abstained": a["abstained"],
               "claims": len(claims),
               "supported": sum(c["verdict"] == "SUPPORTED" for c in claims),
               "partial": sum(c["verdict"] == "PARTIAL" for c in claims),
               "unsupported": sum(c["verdict"] == "UNSUPPORTED" for c in claims),
               "contradicted": sum(c["verdict"] == "CONTRADICTED" for c in claims),
               "weight": sum(WEIGHTS.get(c["verdict"], 0) for c in claims),
               "ev_recall": ev_recall, "relevancy": relev.get(qid)}
        row["hardened"] = 0
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
    ctx_recall = (sum(r["ev_recall"] for r in with_gold) / len(with_gold)
                  if with_gold else None)
    wcer = sum(r["weight"] for r in rows) / n_claims if n_claims else None
    mean_rel = (sum(relev.values()) / len(relev)) if relev else None

    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"  # noqa: E731
    print(f"questions: {len(rows)}  answered: {len(answered)}  "
          f"abstained: {len(rows) - len(answered)}")
    print("RAGAS-style (local stack):")
    print(f"  faithfulness (claim support rate):   {fmt(faith)}")
    print(f"  context recall (vs FinDER gold):     {fmt(ctx_recall)}")
    print(f"  answer relevancy (nomic cosine):     {fmt(mean_rel)}")
    print("DAG-level:")
    print(f"  weighted claim error rate:           {fmt(wcer)}")
    print(f"  hardened-qualifier claims:           {sum(r['hardened'] for r in rows)}")
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
    (out / "index.html").write_text(_dashboard(rows, faith, ctx_recall, mean_rel, wcer))
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
                     ("context recall<br>(vs gold passages)", ctx_recall),
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
