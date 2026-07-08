#!/usr/bin/env python3
"""Deep-dive exhibits: for a handful of questions, render the FULL cited chunks
with every verified quote highlighted in place, plus all claims uncut.

    python tools/provenance_exhibit.py [--ids finder_0020 ...] [--pdf out/provenance/exhibits.pdf]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provenance_report import chunk_texts
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

DEFAULT_IDS = ["finder_0020", "finder_0017", "finder_0002", "placebo_0002"]
VERDICT_COLOR = {"SUPPORTED": "#16a34a", "PARTIAL": "#d97706",
                 "UNSUPPORTED": "#dc2626", "CONTRADICTED": "#991b1b"}
HIGHLIGHT = "#fff3bf"  # verified quote inside the chunk

_SUP = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("TNR", f"{_SUP}/Times New Roman.ttf"))
pdfmetrics.registerFont(TTFont("TNR-Bold", f"{_SUP}/Times New Roman Bold.ttf"))

S = dict(
    h=ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=14, leading=17,
                     spaceAfter=4),
    meta=ParagraphStyle("meta", fontName="Helvetica", fontSize=9.5, leading=13,
                        textColor="#475569", spaceAfter=10),
    sub=ParagraphStyle("sub", fontName="Helvetica-Bold", fontSize=11, leading=14,
                       spaceBefore=10, spaceAfter=4, textColor="#1e293b"),
    body=ParagraphStyle("body", fontName="TNR", fontSize=10.5, leading=14,
                        alignment=TA_JUSTIFY, spaceAfter=6),
    claim=ParagraphStyle("claim", fontName="TNR", fontSize=10.5, leading=14,
                         leftIndent=14, spaceAfter=5),
    chunk=ParagraphStyle("chunk", fontName="TNR", fontSize=9.3, leading=12.6,
                         alignment=TA_JUSTIFY, spaceAfter=8, leftIndent=6,
                         borderPadding=6, backColor="#fafafa"),
)


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def find_dir(qid: str) -> Path | None:
    for root in ("out/provenance", "out/placebo"):
        d = Path(root) / qid
        if (d / "answer.json").exists():
            return d
    return None


def section(qid: str, story: list) -> None:
    d = find_dir(qid)
    if d is None:
        print(f"skip {qid}: no artifacts")
        return
    a = json.loads((d / "answer.json").read_text())
    ver = a["extra"].get("verification", {})
    claims, evidence = ver.get("claims", []), ver.get("evidence", [])
    texts = chunk_texts(d, {e["chunk_id"] for e in evidence})

    story.append(Paragraph(esc(f"{qid} — {a['question']}"), S["h"]))
    outcome = (f"ABSTAINED: {a['abstain_reason']}" if a["abstained"]
               else f"answered, confidence {a['confidence']:.2f}")
    story.append(Paragraph(
        esc(f"{outcome} · model {a['model']} · {len(evidence)} chunks retrieved, "
            f"{len({c['citation'] for c in claims if c['citation']})} cited"), S["meta"]))

    if a["text"]:
        story.append(Paragraph("Answer (full)", S["sub"]))
        story.append(Paragraph(esc(a["text"]).replace("\n", "<br/>"), S["body"]))

    if claims:
        story.append(Paragraph("Claims — verbatim, with judge verdicts", S["sub"]))
        for i, c in enumerate(claims, 1):
            col = VERDICT_COLOR.get(c["verdict"], "#334155")
            flags = "" if c["verdict"] != "SUPPORTED" else (
                " · quote verified" if c["valid_quote"] else " · QUOTE NOT FOUND")
            story.append(Paragraph(
                f'<b>C{i}</b> <font color="{col}"><b>[{c["verdict"]}]</b></font>'
                f'{esc(flags)} — {esc(c["claim"])}'
                + (f'<br/><font color="#475569" size="9">cites chunk '
                   f'[{c["citation"]}]</font>' if c["citation"] else ""), S["claim"]))

    # Full cited chunks with each verified quote highlighted in place.
    cited_idx = sorted({c["citation"] for c in claims if c["citation"]})
    for idx in cited_idx:
        ev = evidence[idx - 1]
        full = texts.get(ev["chunk_id"], "") or ev["quote"]
        marked = esc(full)
        for i, c in enumerate(claims, 1):
            q = (c.get("quote") or "").strip()
            if c["citation"] == idx and q and esc(q) in marked:
                marked = marked.replace(
                    esc(q),
                    f'<font backColor="{HIGHLIGHT}">{esc(q)}</font>'
                    f'<font color="#16a34a" size="8"><b> [C{i}]</b></font>', 1)
        src = (f"Chunk [{idx}] — {ev['ticker']} {ev['filing_type']} "
               f"{ev['filing_date']}, item {ev['item_number']} "
               f"({len(full):,} chars, shown in full) · {ev['chunk_id']}")
        story.append(Paragraph(esc(src), S["sub"]))
        story.append(Paragraph(marked, S["chunk"]))

    uncited = [e for i, e in enumerate(evidence, 1) if i not in cited_idx]
    if uncited:
        story.append(Paragraph(
            esc(f"Retrieved but uncited ({len(uncited)}): ") + "; ".join(
                esc(f"{e['ticker']} {e['filing_type']} {e['filing_date']} "
                    f"it.{e['item_number']}") for e in uncited), S["meta"]))
    story.append(PageBreak())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--pdf", default="out/provenance/exhibits.pdf")
    args = ap.parse_args()

    story: list = []
    for qid in args.ids:
        section(qid, story)
    doc = SimpleDocTemplate(args.pdf, pagesize=letter, leftMargin=0.9 * inch,
                            rightMargin=0.9 * inch, topMargin=0.8 * inch,
                            bottomMargin=0.8 * inch, title="Claim-provenance exhibits")
    doc.build(story)
    print(f"wrote {args.pdf} ({len(args.ids)} exhibits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
