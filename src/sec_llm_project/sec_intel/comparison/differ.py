"""Side-by-side filing comparison (plan item 6).

Compares two filings section-by-section to surface newly introduced disclosures,
removed disclosures, and reworded passages. The matching is deterministic and
dependency-free: each section is broken into "units" (paragraph/sentence sized),
and units are aligned across filings by token Jaccard similarity.

Typical uses: a 10-K vs the prior year's 10-K, or a 10-Q vs the previous
quarter, with special attention to Risk Factors (Item 1A).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.types import Chunk
from ..index.store import SECIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Similarity thresholds for aligning disclosure units across filings.
_SAME = 0.65   # >= this: effectively unchanged
_CHANGED = 0.30  # in [_CHANGED, _SAME): reworded/material change


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_units(text: str, min_chars: int = 80) -> list[str]:
    """Break a section into disclosure-sized units (roughly one claim each).

    Splits on sentence boundaries, then merges consecutive short fragments up to
    ``min_chars`` so each unit is comparable across filings. This granularity is
    what lets the differ flag an individual newly added risk factor.
    """
    fragments: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para_text = para.strip()
        if not para_text:
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n+", para_text):
            sent_text = sent.strip()
            if sent_text:
                fragments.append(sent_text)

    units: list[str] = []
    buf = ""
    for frag in fragments:
        buf = f"{buf} {frag}".strip() if buf else frag
        if len(buf) >= min_chars:
            units.append(buf)
            buf = ""
    if buf:
        if units and len(buf) < min_chars:
            units[-1] = f"{units[-1]} {buf}"
        else:
            units.append(buf)
    return units or ([text.strip()] if text.strip() else [])


@dataclass
class SectionDiff:
    item_number: str
    section_title: str
    similarity: float
    added: list[str] = field(default_factory=list)      # in B only (new disclosures)
    removed: list[str] = field(default_factory=list)     # in A only (removed)
    changed: list[tuple[str, str]] = field(default_factory=list)  # (old, new) rewordings

    def to_dict(self) -> dict[str, object]:
        return {
            "item_number": self.item_number,
            "section_title": self.section_title,
            "similarity": round(self.similarity, 4),
            "added": self.added,
            "removed": self.removed,
            "changed": [{"old": o, "new": n} for o, n in self.changed],
        }


@dataclass
class ComparisonReport:
    ticker: str
    label_a: str
    label_b: str
    sections: list[SectionDiff] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "filing_a": self.label_a,
            "filing_b": self.label_b,
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, int]:
        return {
            "sections_compared": len(self.sections),
            "added": sum(len(s.added) for s in self.sections),
            "removed": sum(len(s.removed) for s in self.sections),
            "changed": sum(len(s.changed) for s in self.sections),
        }


def _section_text(chunks: list[Chunk], item_number: str) -> tuple[str, str]:
    items = [c for c in chunks if c.item_number == item_number]
    items.sort(key=lambda c: c.char_start)
    text = "\n\n".join(c.text for c in items)
    title = items[0].section_title if items else f"Item {item_number}"
    return text, title


def diff_section(item_number: str, chunks_a: list[Chunk], chunks_b: list[Chunk]) -> SectionDiff:
    text_a, title_a = _section_text(chunks_a, item_number)
    text_b, title_b = _section_text(chunks_b, item_number)
    units_a = _split_units(text_a)
    units_b = _split_units(text_b)
    toks_a = [_tokens(u) for u in units_a]
    toks_b = [_tokens(u) for u in units_b]

    matched_a: set[int] = set()
    added: list[str] = []
    changed: list[tuple[str, str]] = []

    for j, ub in enumerate(units_b):
        best_i, best_sim = -1, 0.0
        for i, _ta in enumerate(toks_a):
            if i in matched_a:
                continue
            sim = _jaccard(toks_b[j], _ta)
            if sim > best_sim:
                best_i, best_sim = i, sim
        if best_sim >= _SAME and best_i >= 0:
            matched_a.add(best_i)
        elif best_sim >= _CHANGED and best_i >= 0:
            matched_a.add(best_i)
            changed.append((units_a[best_i], ub))
        else:
            added.append(ub)

    removed = [units_a[i] for i in range(len(units_a)) if i not in matched_a]
    overall = _jaccard(_tokens(text_a), _tokens(text_b))
    return SectionDiff(
        item_number=item_number, section_title=title_b or title_a,
        similarity=overall, added=added, removed=removed, changed=changed,
    )


class FilingComparer:
    def __init__(self, index: SECIndex) -> None:
        self.index = index

    def _filing_chunks(self, ticker: str, *, accession: str | None = None,
                       fiscal_year: str | None = None,
                       filing_type: str | None = None) -> list[Chunk]:
        out: list[Chunk] = []
        for c in self.index.all_chunks():
            m = c.metadata
            if m.ticker.upper() != ticker.upper():
                continue
            if accession and m.accession != accession:
                continue
            if fiscal_year and m.fiscal_year != str(fiscal_year):
                continue
            if filing_type and m.filing_type != filing_type:
                continue
            out.append(c)
        return out

    def compare(self, chunks_a: list[Chunk], chunks_b: list[Chunk], *,
                ticker: str = "UNKNOWN", label_a: str = "A", label_b: str = "B",
                items: list[str] | None = None) -> ComparisonReport:
        present = items or sorted(
            {c.item_number for c in chunks_a} | {c.item_number for c in chunks_b}
        )
        report = ComparisonReport(ticker=ticker, label_a=label_a, label_b=label_b)
        for item in present:
            if item in {"UNKNOWN", "0"}:
                continue
            report.sections.append(diff_section(item, chunks_a, chunks_b))
        return report

    def compare_years(self, ticker: str, year_a: str, year_b: str, *,
                      filing_type: str = "10-K",
                      items: list[str] | None = None) -> ComparisonReport:
        chunks_a = self._filing_chunks(ticker, fiscal_year=str(year_a), filing_type=filing_type)
        chunks_b = self._filing_chunks(ticker, fiscal_year=str(year_b), filing_type=filing_type)
        return self.compare(
            chunks_a, chunks_b, ticker=ticker,
            label_a=f"{filing_type} {year_a}", label_b=f"{filing_type} {year_b}",
            items=items,
        )
