#!/usr/bin/env python3
"""Build the charts of docs/REPORT.md as standalone SVGs (no plotting dependency).

Each SVG carries its own light and dark colors (prefers-color-scheme), so it
follows the reader's GitHub theme. End-to-end numbers are read from the run
directories under out/; retrieval-lab numbers are the ones recorded in
docs/EXPERIMENTS.md (section noted next to each constant).

    python tools/make_report_figures.py            # writes docs/figures/*.svg
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "out"

# Categorical slots in fixed order (light, dark); text and grid tokens.
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
          ("#eda100", "#c98500"), ("#e87ba4", "#d55181")]
MUTED = ("#b9b8b2", "#5c5b57")  # de-emphasized marks (comparison bars)

CSS = """
svg { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.bg { fill: #ffffff; } .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; } .t3 { fill: #7a7974; }
.grid { stroke: #e4e3df; stroke-width: 1; } .axis { stroke: #b9b8b2; stroke-width: 1; }
.gap { stroke: #ffffff; stroke-width: 2; }
%s
@media (prefers-color-scheme: dark) {
  .bg { fill: #0d1117; } .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; } .t3 { fill: #9a998f; }
  .grid { stroke: #2a2a28; } .axis { stroke: #5c5b57; } .gap { stroke: #0d1117; }
  %s
}
"""


def _css() -> str:
    light = " ".join(f".s{i} {{ fill: {lt}; }}" for i, (lt, _) in enumerate(SERIES))
    dark = " ".join(f".s{i} {{ fill: {dk}; }}" for i, (_, dk) in enumerate(SERIES))
    light += f" .mut {{ fill: {MUTED[0]}; }}"
    dark += f" .mut {{ fill: {MUTED[1]}; }}"
    return CSS % (light, dark)


class Svg:
    def __init__(self, w: int, h: int, title: str, desc: str) -> None:
        self.w, self.h = w, h
        self.parts = [f'<rect class="bg" width="{w}" height="{h}"/>']
        self.title, self.desc = title, desc

    def text(self, x, y, s, cls="t2", size=13, anchor="start", weight="normal"):  # noqa: PLR0917
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>')

    def line(self, x1, y1, x2, y2, cls="grid"):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'class="{cls}"/>')

    def rect(self, x, y, w, h, cls, r=0, tip=""):  # noqa: PLR0917
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" '
                          f'height="{h:.1f}" rx="{r}" class="{cls}">{t}</rect>')

    def hbar(self, x, y, w, h, cls, tip=""):  # noqa: PLR0917
        """Horizontal bar: square at the baseline, 4px rounded data end."""
        r = min(4, h / 2, w / 2)
        p = (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
             f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} "
             f"H{x:.1f} Z")
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.parts.append(f'<path d="{p}" class="{cls}">{t}</path>')

    def vbar(self, x, y0, w, h, cls, tip=""):  # noqa: PLR0917
        """Vertical bar from baseline y0 up by h, 4px rounded top."""
        r = min(4, w / 2, h / 2) if h > 0 else 0
        y = y0 - h
        p = (f"M{x:.1f},{y0:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
             f"H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y0:.1f} Z")
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.parts.append(f'<path d="{p}" class="{cls}">{t}</path>')

    def write(self, name: str) -> None:
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}" role="img" aria-labelledby="t d">\n'
               f'<title id="t">{escape(self.title)}</title><desc id="d">{escape(self.desc)}</desc>\n'
               f"<style>{_css()}</style>\n{body}\n</svg>\n")
        FIG.mkdir(parents=True, exist_ok=True)
        (FIG / name).write_text(svg)
        print("wrote", FIG / name)


# ----------------------------------------------------------------- 1. retrieval
# FinDER dev, 122 scorable questions, Hit@8 (EXPERIMENTS.md §1, §6, §7.2).
RETRIEVAL = [
    ("Original query instruction", 0.361, "§1"),
    ("Fixed query instruction", 0.525, "§1"),
    ("+ MMR diversity (λ 0.85)", 0.590, "§6"),
    ("+ company-blind HyDE rewrite", 0.787, "§7.2"),
]


def fig_retrieval() -> None:
    s = Svg(760, 300, "Retrieval Hit@8 on FinDER dev, step by step",
            "Share of 122 scorable FinDER dev questions with a gold chunk in the top 8: "
            + "; ".join(f"{n} {v:.3f}" for n, v, _ in RETRIEVAL))
    s.text(24, 32, "Retrieval: share of questions with the gold passage in the top 8",
           "t1", 16, weight="600")
    s.text(24, 54, "FinDER dev, 122 scorable questions; each step adds to the one above",
           "t3", 12)
    x0, x1, top, bh, gap = 250, 690, 80, 34, 16
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + v * (x1 - x0)
        s.line(x, top - 6, x, top + 4 * (bh + gap) - gap + 6)
        s.text(x, top + 4 * (bh + gap) + 10, f"{v:.0%}" if v else "0", "t3", 11, "middle")
    for i, (name, v, sec) in enumerate(RETRIEVAL):
        y = top + i * (bh + gap)
        s.text(x0 - 12, y + bh / 2 + 5, name, "t2", 13, "end")
        cls = "s0" if i == len(RETRIEVAL) - 1 else "mut"
        s.hbar(x0, y, v * (x1 - x0), bh, cls, f"{name}: Hit@8 {v:.3f} (EXPERIMENTS.md {sec})")
        s.text(x0 + v * (x1 - x0) + 8, y + bh / 2 + 5, f"{v:.2f}", "t1", 13, weight="600")
    s.write("retrieval_hit8.svg")


# ------------------------------------------------------------ 2. error sources
RUNS = [("v1", "provenance_16gb"), ("v2", "provenance_16gb_v2"), ("v3", "provenance_16gb_v3"),
        ("v3 + gold", "provenance_16gb_v3_gold"), ("v4", "provenance_16gb_v4")]
GROUPS = [
    ("Clean", ["clean"]),
    ("Dropped qualifier", ["generation: dropped qualifiers"]),
    ("Unsupported claims", ["generation: unsupported claims"]),
    ("Abstained with evidence", ["abstained (gate/judge)"]),
    ("Retrieval miss / wrong period", ["retrieval miss -> abstained",
                                       "retrieval miss -> ungrounded claims",
                                       "wrong-period evidence"]),
]


def error_counts() -> list[tuple[str, list[int]]]:
    rows = []
    for label, d in RUNS:
        qs = [r for r in _csv(OUT / d / "questions.csv") if r["id"].startswith("finder_")]
        c = Counter(r["error_source"] for r in qs)
        counts = [sum(c[k] for k in keys) for _, keys in GROUPS]
        assert sum(counts) == len(qs) == 260, (label, c)
        rows.append((label, counts))
    return rows


def fig_errors(rows) -> None:
    s = Svg(820, 372, "Where the 260 FinDER questions end up, by version",
            "Main outcome per FinDER question for each end-to-end run: "
            + "; ".join(f"{lab}: " + ", ".join(f"{g[0]} {n}" for g, n in zip(GROUPS, c,
                                                                                strict=True))
                        for lab, c in rows))
    s.text(24, 32, "End to end: main outcome of each of the 260 FinDER questions", "t1", 16,
           weight="600")
    s.text(24, 54, "One label per question (its dominant failure); gemma4:12b answers, "
                   "qwen3:14b judges", "t3", 12)
    # legend, one row above the plot
    lx = 24
    for i, (g, _) in enumerate(GROUPS):
        s.rect(lx, 70, 12, 12, f"s{i}", 2)
        s.text(lx + 18, 81, g, "t2", 12)
        lx += 18 + 6.4 * len(g) + 20
    x0, x1, top, bh, gap = 110, 760, 104, 34, 14
    for v in (0, 65, 130, 195, 260):
        x = x0 + v / 260 * (x1 - x0)
        s.line(x, top - 6, x, top + len(rows) * (bh + gap) - gap + 6)
        s.text(x, top + len(rows) * (bh + gap) + 10, str(v), "t3", 11, "middle")
    for i, (lab, counts) in enumerate(rows):
        y = top + i * (bh + gap)
        s.text(x0 - 12, y + bh / 2 + 5, lab, "t2", 13, "end")
        x = x0
        for j, n in enumerate(counts):
            w = n / 260 * (x1 - x0)
            s.rect(x, y, w, bh, f"s{j}", 0, f"{lab} - {GROUPS[j][0]}: {n} of 260")
            if j:
                s.line(x, y, x, y + bh, "gap")
            if j == 0:
                s.text(x + w - 8, y + bh / 2 + 5, str(n), "bg", 13, "end", "600")
            x += w
    s.write("error_sources.svg")


# --------------------------------------------------------- 3. answer stage (56)
STAGE_RUNS = [("v3*", "provenance_16gb_v3_gold"), ("v4 (qualifier rule)", "provenance_16gb_v4"),
              ("v5 (+ brevity)", "provenance_16gb_v5_qualsub")]
# Hand-labelled real dropped qualifiers on the same 56 questions (eval/qualifier_labels*.json).
# * The 56 were picked from v3's lexical flags, so v3 itself has 0 clean answers on them
#   by selection; words and clean come from the independent v3_gold run, and the dropped
#   qualifiers from the v3 hand labels (the v3_gold run was not hand-labelled).
REAL_DROPS = {"provenance_16gb_v3_gold": "eval/qualifier_labels.json", "provenance_16gb_v4": "eval/qualifier_labels_v4.json",
              "provenance_16gb_v5_qualsub": "eval/qualifier_labels_v5.json"}


def _json(path: str | Path):
    return json.loads((ROOT / path).read_text())


def _csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def stage_numbers() -> list[dict]:
    ids = {c["id"] for c in _json("eval/qualifier_subset.json")}
    out = []
    for label, d in STAGE_RUNS:
        words, unsup, claims = [], 0, 0
        for i in ids:
            p = OUT / d / i / "answer.json"
            if not p.exists():
                continue
            a = json.loads(p.read_text())
            if a.get("abstained"):
                continue
            words.append(len(a["text"].split()))
            cl = a.get("extra", {}).get("verification", {}).get("claims", [])
            claims += len(cl)
            unsup += sum(c.get("verdict") == "UNSUPPORTED" for c in cl)
        rows = [r for r in _csv(OUT / d / "questions.csv") if r["id"] in ids]
        real = sum(x["real"] for x in _json(REAL_DROPS[d])["items"] if x["id"] in ids)
        out.append({"label": label, "words": statistics.mean(words), "real": real,
                    "clean": sum(r["error_source"] == "clean" for r in rows),
                    "unsup": unsup / claims})
    return out


def fig_stage(nums) -> None:
    panels = [("Words per answer", "words", "{:.0f}"),
              ("Real dropped qualifiers", "real", "{:d}"),
              ("Clean answers (of 56)", "clean", "{:d}")]
    s = Svg(820, 344, "Answer-stage changes on the same 56 questions",
            "; ".join(f"{n['label']}: {n['words']:.0f} words per answer, {n['real']} real dropped "
                      f"qualifiers, {n['clean']} clean" for n in nums))
    s.text(24, 32, "Answer stage: same 56 questions, three prompt versions", "t1", 16,
           weight="600")
    s.text(24, 54, "Questions where v3 dropped hedges; dropped qualifiers hand-labelled "
                   "(eval/qualifier_labels*.json)", "t3", 12)
    s.text(24, 316, "* v3: words and clean answers from the v3 + gold-passages run; the 56 were "
                    "picked from v3's flags,", "t3", 11)
    s.text(24, 330, "  so v3 itself has 0 clean answers on them by selection.", "t3", 11)
    pw, base, ph = 250, 270, 160
    for k, (title, key, fmt) in enumerate(panels):
        px = 24 + k * (pw + 20)
        s.text(px, 88, title, "t1", 13, weight="600")
        vmax = max(n[key] for n in nums) * 1.15
        s.line(px, base, px + pw - 20, base, "axis")
        bw = 52
        for j, n in enumerate(nums):
            x = px + 14 + j * (bw + 26)
            h = n[key] / vmax * ph
            cls = "s0" if j == len(nums) - 1 else "mut"
            s.vbar(x, base, bw, h, cls, f"{n['label']}: {title.lower()} {fmt.format(n[key])}")
            s.text(x + bw / 2, base - h - 8, fmt.format(n[key]), "t1", 13, "middle", "600")
            for li, part in enumerate(n["label"].split(" (")):
                s.text(x + bw / 2, base + 18 + li * 14,
                       part if li == 0 else "(" + part, "t2" if li == 0 else "t3",
                       11 if li == 0 else 10, "middle")
    s.write("answer_stage.svg")


# ---------------------------------------------------------------- 4. rerankers
# v3 FinDER, 249 scorable, Hit@8 (EXPERIMENTS.md §13).
RERANK = [("Production: dense + HyDE + MMR", 0.775), ("RRF with Qwen3-Reranker", 0.675),
          ("RRF with MiniLM", 0.671), ("Qwen3-Reranker, then MMR", 0.655),
          ("Qwen3-Reranker", 0.635), ("RRF with bge-reranker-v2-m3", 0.635),
          ("MiniLM, then MMR", 0.562), ("MiniLM", 0.542), ("bge-reranker-v2-m3, then MMR", 0.534),
          ("bge-reranker-v2-m3", 0.518)]


def fig_rerank() -> None:
    s = Svg(760, 92 + len(RERANK) * 30 + 30, "Cross-encoder rerankers vs production retrieval",
            "Hit@8 on 249 scorable FinDER questions: " + "; ".join(f"{n} {v:.3f}" for n, v in RERANK))
    s.text(24, 32, "Every reranker lowered Hit@8 (so none is used)", "t1", 16, weight="600")
    s.text(24, 54, "FinDER, 249 scorable questions, v3 index; all differences have 95% CIs "
                   "below zero", "t3", 12)
    x0, x1, top, bh, gap = 250, 690, 76, 20, 10
    n = len(RERANK)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + v * (x1 - x0)
        s.line(x, top - 6, x, top + n * (bh + gap) - gap + 6)
        s.text(x, top + n * (bh + gap) + 10, f"{v:.0%}" if v else "0", "t3", 11, "middle")
    for i, (name, v) in enumerate(RERANK):
        y = top + i * (bh + gap)
        s.text(x0 - 12, y + bh / 2 + 4, name, "t1" if i == 0 else "t2", 12, "end",
               "600" if i == 0 else "normal")
        s.hbar(x0, y, v * (x1 - x0), bh, "s0" if i == 0 else "mut", f"{name}: Hit@8 {v:.3f}")
        s.text(x0 + v * (x1 - x0) + 8, y + bh / 2 + 4, f"{v:.2f}", "t1", 12)
    s.write("rerankers.svg")


def main() -> int:
    fig_retrieval()
    rows = error_counts()
    fig_errors(rows)
    for lab, c in rows:
        print(lab, dict(zip([g for g, _ in GROUPS], c, strict=True)))
    nums = stage_numbers()
    fig_stage(nums)
    for n in nums:
        print(n)
    fig_rerank()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
