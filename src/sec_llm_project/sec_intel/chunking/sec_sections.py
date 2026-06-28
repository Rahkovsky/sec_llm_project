"""Section-aware chunking for SEC filings.

Rather than slicing on fixed token windows, filings are first split on their
canonical ``Item N.`` boundaries (e.g. *Item 1A. Risk Factors*). Each section is
then divided into overlapping character windows on sentence/paragraph
boundaries. Every resulting chunk carries the section title and item number, so
downstream citations can name exactly where a passage came from (plan item 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..core.config import ChunkingConfig
from ..core.types import Chunk, FilingMetadata, make_chunk_id

# Canonical 10-K item titles, used to label detected sections.
ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
}

# Matches "Item 1A." / "ITEM 7." at the start of a line, optionally followed by
# a title on the same line.
_ITEM_RE = re.compile(
    r"^\s*ITEM\s+(?P<num>\d{1,2}[A-C]?)\s*[\.\:\-—]?\s*(?P<title>[^\n]{0,80})",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Section:
    item_number: str
    title: str
    text: str
    char_start: int


def metadata_from_filename(path: str | Path, *, filing_type: str = "10-K") -> FilingMetadata:
    """Best-effort provenance from the downloader's ``TICKER_ACCESSION.txt`` naming.

    Supports both the modern ``AMZN_0001018724-25-000004`` accession pattern and
    legacy ``TICKER_CIK_DATE_ACCESSION`` layouts.
    """
    p = Path(path)
    stem = p.stem
    parts = stem.split("_")
    ticker = parts[0].upper() if parts and parts[0] else "UNKNOWN"
    cik = "UNKNOWN"
    accession = "UNKNOWN"
    filing_date = "UNKNOWN"
    fiscal_year = "UNKNOWN"

    if len(parts) >= 2:
        rest = parts[1]
        m = re.match(r"(?P<cik>\d{10})-(?P<yy>\d{2})-(?P<seq>\d{6})$", rest)
        if m:
            cik = m.group("cik")
            accession = f"{m.group('cik')}-{m.group('yy')}-{m.group('seq')}"
            fiscal_year = f"20{m.group('yy')}"
            filing_date = fiscal_year
        else:
            cik = rest
    if len(parts) >= 3:
        filing_date = parts[2]
        fiscal_year = parts[2][:4] if parts[2][:4].isdigit() else fiscal_year
    if len(parts) >= 4:
        accession = "_".join(parts[3:])

    return FilingMetadata(
        ticker=ticker, company="UNKNOWN", cik=cik, filing_type=filing_type,
        filing_date=filing_date, fiscal_year=fiscal_year, accession=accession,
        source_path=str(p),
    )


def split_into_sections(text: str) -> list[Section]:
    """Split a filing into ``Item N.`` sections; fall back to one whole-document section."""
    matches = list(_ITEM_RE.finditer(text))
    # Keep only the last occurrence per item number to skip the table-of-contents
    # listing that repeats every item near the top of a filing.
    last_by_item: dict[str, re.Match[str]] = {}
    for m in matches:
        num = m.group("num").upper()
        last_by_item[num] = m
    ordered = sorted(last_by_item.values(), key=lambda m: m.start())

    if not ordered:
        return [Section(item_number="UNKNOWN", title="Full Document", text=text, char_start=0)]

    sections: list[Section] = []
    preamble = text[: ordered[0].start()].strip()
    if len(preamble) > 200:
        sections.append(Section("0", "Cover / Front Matter", preamble, 0))

    for i, m in enumerate(ordered):
        num = m.group("num").upper()
        start = m.start()
        end = ordered[i + 1].start() if i + 1 < len(ordered) else len(text)
        body = text[start:end].strip()
        title = ITEM_TITLES.get(num) or (m.group("title").strip().title() or f"Item {num}")
        sections.append(Section(item_number=num, title=title, text=body, char_start=start))
    return sections


def sections_from_sidecar(sidecar: dict, text: str) -> list[Section]:
    """Rebuild :class:`Section` objects from a download-time sidecar.

    Lets the indexer reuse the segmentation computed once at ingest instead of
    re-splitting on every build. Offsets index into ``text`` (the saved filing).
    """
    sections: list[Section] = []
    for s in sidecar.get("sections", []):
        start = int(s.get("char_start", 0))
        end = int(s.get("char_end", len(text)))
        num = str(s.get("item_number", "UNKNOWN"))
        sections.append(
            Section(
                item_number=num,
                title=str(s.get("title") or f"Item {num}"),
                text=text[start:end].strip(),
                char_start=start,
            )
        )
    return sections


def _window_split(text: str, max_chars: int, overlap: int, min_chars: int) -> list[tuple[int, str]]:
    """Greedy paragraph/sentence-aware windowing. Returns (offset, text) pairs."""
    text = text.strip()
    if len(text) <= max_chars:
        return [(0, text)] if text else []

    # Prefer to break on paragraph then sentence boundaries near the window edge.
    boundaries = [m.end() for m in re.finditer(r"(\n\n|(?<=[.!?])\s+)", text)]
    windows: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        target = start + max_chars
        if target >= n:
            chunk = text[start:n].strip()
            if chunk:
                windows.append((start, chunk))
            break
        # Cut at the latest paragraph/sentence boundary within the window.
        cut = next((b for b in reversed(boundaries) if start < b <= target), target)
        chunk = text[start:cut].strip()
        if chunk:
            windows.append((start, chunk))
        # Advance with overlap, but always make forward progress.
        next_start = cut - overlap
        start = next_start if next_start > start else cut
    # Merge a trailing too-small window into the previous one.
    if len(windows) >= 2 and len(windows[-1][1]) < min_chars:
        prev_off, prev_txt = windows[-2]
        windows[-2] = (prev_off, prev_txt + " " + windows[-1][1])
        windows.pop()
    return windows


class SECChunker:
    """Turns a filing's full text into a list of provenance-rich :class:`Chunk`."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_filing(self, text: str, metadata: FilingMetadata,
                     *, sections: list[Section] | None = None) -> list[Chunk]:
        cfg = self.config
        if sections is None:
            sections = (
                split_into_sections(text)
                if cfg.sec_aware
                else [Section("UNKNOWN", "Full Document", text, 0)]
            )
        chunks: list[Chunk] = []
        for section in sections:
            windows = _window_split(
                section.text, cfg.max_chars, cfg.overlap_chars, cfg.min_chars
            )
            for ordinal, (offset, body) in enumerate(windows):
                if len(body) < cfg.min_chars and len(windows) > 1:
                    continue
                cid = make_chunk_id(metadata, section.item_number, len(chunks))
                abs_start = section.char_start + offset
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        text=body,
                        metadata=metadata,
                        section_title=section.title,
                        item_number=section.item_number,
                        char_start=abs_start,
                        char_end=abs_start + len(body),
                        extra={"section_ordinal": ordinal},
                    )
                )
        return chunks
