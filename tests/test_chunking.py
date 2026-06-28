from __future__ import annotations

from sec_llm_project.sec_intel.chunking.sec_sections import (
    SECChunker,
    metadata_from_filename,
    split_into_sections,
)
from sec_llm_project.sec_intel.core.config import ChunkingConfig
from sec_llm_project.sec_intel.core.types import FilingMetadata

FILING = """Item 1. Business
We make widgets.

Item 1A. Risk Factors
Competition is intense. Supply chains may fail.

Item 7. Management's Discussion and Analysis
Revenue rose.
"""


def test_split_into_sections_detects_items():
    sections = split_into_sections(FILING)
    items = {s.item_number for s in sections}
    assert {"1", "1A", "7"} <= items
    titles = {s.item_number: s.title for s in sections}
    assert titles["1A"] == "Risk Factors"


def test_chunk_metadata_preserved():
    chunker = SECChunker(ChunkingConfig(max_chars=200, overlap_chars=20, min_chars=10))
    meta = FilingMetadata(ticker="ABC", filing_type="10-K", fiscal_year="2024")
    chunks = chunker.chunk_filing(FILING, meta)
    assert chunks
    risk = [c for c in chunks if c.item_number == "1A"]
    assert risk
    assert risk[0].section_title == "Risk Factors"
    assert risk[0].metadata.ticker == "ABC"
    # chunk ids are deterministic
    again = chunker.chunk_filing(FILING, meta)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]


def test_metadata_from_filename_modern_accession():
    meta = metadata_from_filename("data/AMZN_0001018724-25-000004.txt")
    assert meta.ticker == "AMZN"
    assert meta.accession == "0001018724-25-000004"
    assert meta.fiscal_year == "2025"


def test_toc_dedup_picks_last_item_occurrence():
    text = "Item 1A. Risk Factors\n\n" + FILING  # a fake table-of-contents line up top
    sections = split_into_sections(text)
    risk = [s for s in sections if s.item_number == "1A"]
    assert len(risk) == 1
    assert "Competition is intense" in risk[0].text
