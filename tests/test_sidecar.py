"""Provenance sidecar: rich metadata + precomputed sections flow into chunks,
with a graceful fallback to filename-derived metadata when no sidecar exists.
"""

from __future__ import annotations

import json

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.index.builder import IndexBuilder

_FILING = """\
UNITED STATES SECURITIES AND EXCHANGE COMMISSION

Item 1. Business
Nova Devices designs and sells precision sensors to industrial customers.

Item 1A. Risk Factors
Our supply chain is concentrated in a single region and any disruption could
materially harm our results. We also face significant cybersecurity risks.

Item 3. Legal Proceedings
We are subject to a putative class action filed in 2024 alleging product defects.
"""


def _write_filing(dir_path, *, with_sidecar: bool):
    dir_path.mkdir(parents=True, exist_ok=True)
    txt = dir_path / "NOVA_0000999999-25-000001.txt"
    txt.write_text(_FILING, encoding="utf-8")
    if with_sidecar:
        sidecar = {
            "schema_version": 1,
            "metadata": {
                "ticker": "NOVA", "company": "Nova Devices Inc.", "cik": "999999",
                "filing_type": "10-K", "filing_date": "2025-02-14",
                "fiscal_year": "2024", "accession": "0000999999-25-000001",
                "source_url": (
                    "https://www.sec.gov/Archives/edgar/data/999999/"
                    "000099999925000001/0000999999-25-000001-index.htm"
                ),
                "source_path": str(txt),
            },
            "sections": [
                {"item_number": "1", "title": "Business",
                 "char_start": _FILING.index("Item 1."), "char_end": _FILING.index("Item 1A.")},
                {"item_number": "1A", "title": "Risk Factors",
                 "char_start": _FILING.index("Item 1A."), "char_end": _FILING.index("Item 3.")},
                {"item_number": "3", "title": "Legal Proceedings",
                 "char_start": _FILING.index("Item 3."), "char_end": len(_FILING)},
            ],
        }
        txt.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return txt


def test_sidecar_metadata_flows_into_chunks(tmp_path):
    _write_filing(tmp_path / "10-K" / "NOVA", with_sidecar=True)
    index = IndexBuilder(AppConfig()).build_from_dir(tmp_path / "10-K", persist=False)
    chunks = index.all_chunks()

    assert chunks
    # Rich provenance comes from the sidecar, not guessed from the filename.
    assert all(c.metadata.company == "Nova Devices Inc." for c in chunks)
    assert all(c.metadata.source_url.startswith("https://www.sec.gov/") for c in chunks)
    assert all(c.metadata.filing_date == "2025-02-14" for c in chunks)
    # Segmentation from the sidecar is preserved.
    assert {c.item_number for c in chunks} >= {"1", "1A", "3"}


def test_missing_sidecar_falls_back_to_filename(tmp_path):
    _write_filing(tmp_path / "10-K" / "NOVA", with_sidecar=False)
    index = IndexBuilder(AppConfig()).build_from_dir(tmp_path / "10-K", persist=False)
    chunks = index.all_chunks()

    assert chunks  # still indexes via metadata_from_filename + live splitting
    assert all(c.metadata.ticker == "NOVA" for c in chunks)
    # No sidecar => no source URL available.
    assert all(c.metadata.source_url == "" for c in chunks)
