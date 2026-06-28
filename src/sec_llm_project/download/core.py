#!/usr/bin/env python3
"""Streamlined SEC filing downloads using edgartools."""

import json
import os
import re
from typing import Any, cast

from edgar import Company, set_identity

from sec_llm_project.download.constants import DEFAULT_UA, FORM_TYPES, OUTPUT_BASE
from sec_llm_project.utils.env_config import get_sec_user_info
from sec_llm_project.utils.logging_config import get_logger


class SECExtractor:
    """Downloads and extracts 10-K/20-F filings."""

    def __init__(self, user_agent: str = DEFAULT_UA, output_dir: str = OUTPUT_BASE):
        self.output_dir = output_dir
        self.logger = get_logger(__name__)

        if "@" in user_agent:
            email = user_agent.rsplit(maxsplit=1)[-1]
        else:
            _, email = get_sec_user_info()
        set_identity(email)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean up text formatting."""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _get_filing(
        self,
        ticker: str,
        forms: list[str] | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[object | None, str | None]:
        """Get filing, optionally filtered by forms, year, and date range."""
        co = Company(ticker)
        if not forms:
            forms = FORM_TYPES

        # Try each form type
        for form in forms:
            filings = cast(Any, co).get_filings(form=form)
            if filings:
                if year or start_date or end_date:
                    for filing in filings:  # Filter by date criteria
                        if getattr(filing, "filing_date", None):
                            if (
                                year and filing.filing_date.year != year
                            ):  # Check year filter
                                continue
                            if start_date:  # Check date range filters
                                from datetime import datetime

                                start = datetime.strptime(start_date, "%Y-%m-%d").date()
                                if filing.filing_date < start:
                                    continue
                            if end_date:
                                from datetime import datetime

                                end = datetime.strptime(end_date, "%Y-%m-%d").date()
                                if filing.filing_date > end:
                                    continue

                            return filing, form
                else:
                    return filings.latest(), form

        return None, None

    @staticmethod
    def _first_attr(obj: object, *names: str) -> Any:
        """First truthy attribute among ``names`` (defensive across edgartools versions).

        Uses a try/except rather than getattr's default so that properties that
        raise (e.g. TypeError when edgartools can't parse a malformed filing) are
        treated as missing rather than crashing.
        """
        for name in names:
            try:
                value = getattr(obj, name)
                if value:
                    return value
            except Exception:
                continue
        return None

    @classmethod
    def _source_url(cls, filing: object, cik: str, accession: str) -> str:
        """Filing's EDGAR URL, from edgartools or constructed from CIK + accession."""
        url = cls._first_attr(filing, "filing_url", "url", "homepage_url")
        if url:
            return str(url)
        if cik not in ("", "UNKNOWN") and accession:
            try:
                cik_int = int(cik)
            except (TypeError, ValueError):
                return ""
            nodash = accession.replace("-", "")
            return (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{nodash}/"
                f"{accession}-index.htm"
            )
        return ""

    def _capture_metadata(self, filing: object, ticker: str, form_type: str,
                          source_path: str) -> dict[str, str]:
        """Capture full provenance from the edgartools filing at download time.

        The legacy path re-derived metadata from the filename, losing the company
        name, real filing date, and source URL. All of it is in hand here, so we
        persist it into a sidecar for the indexer to consume verbatim.
        """
        try:
            accession = str(filing.accession_no or "")  # type: ignore[union-attr]
        except Exception:
            accession = ""
        cik_val = self._first_attr(filing, "cik")
        cik = str(cik_val) if cik_val is not None else "UNKNOWN"
        company = self._first_attr(filing, "company", "company_name", "name")
        try:
            fdate = filing.filing_date  # type: ignore[union-attr]
        except Exception:
            fdate = None
        filing_date = (
            fdate.isoformat() if hasattr(fdate, "isoformat")
            else (str(fdate) if fdate else "UNKNOWN")
        )
        period = self._first_attr(filing, "period_of_report")
        if period and hasattr(period, "year"):
            fiscal_year = str(period.year)
        elif period:
            fiscal_year = str(period)[:4]
        elif filing_date != "UNKNOWN":
            fiscal_year = filing_date[:4]
        else:
            fiscal_year = "UNKNOWN"
        return {
            "ticker": ticker,
            "company": str(company) if company else ticker,
            "cik": cik,
            "filing_type": str(self._first_attr(filing, "form", "form_type") or form_type or "UNKNOWN"),
            "filing_date": filing_date,
            "fiscal_year": fiscal_year,
            "accession": accession,
            "source_url": self._source_url(filing, cik, accession),
            "source_path": source_path,
        }

    def _write_sidecar(self, txt_path: str, text: str, metadata: dict[str, str]) -> None:
        """Write a ``<stem>.json`` provenance + segmentation sidecar beside the filing.

        Sections are split once here (at ingest) rather than on every index build;
        offsets index into ``txt_path`` so a citation can locate the exact span.
        Atomic via a temp file; a no-op if the sidecar already exists.
        """
        sidecar_path = os.path.splitext(txt_path)[0] + ".json"
        if os.path.exists(sidecar_path):
            return
        from sec_llm_project.sec_intel.chunking.sec_sections import split_into_sections

        secs = split_into_sections(text)
        sections = [
            {
                "item_number": s.item_number,
                "title": s.title,
                "char_start": s.char_start,
                "char_end": secs[i + 1].char_start if i + 1 < len(secs) else len(text),
            }
            for i, s in enumerate(secs)
        ]
        payload = {"schema_version": 1, "metadata": metadata, "sections": sections}
        tmp = sidecar_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, sidecar_path)

    def download_and_extract(
        self,
        ticker: str,
        forms: list[str] | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> int:
        """Download and extract filing. Returns 1 for success, 0 for failure."""
        self.logger.info(f"Processing {ticker}")
        try:  # Get filing info (external API call - can fail)
            filing, form_type = self._get_filing(
                ticker, forms, year, start_date, end_date
            )
        except Exception as e:  # edgartools/network errors
            self.logger.error(f"Failed to fetch filing data for {ticker}: {e}")
            return 0

        if not filing:
            self.logger.warning(f"No 10-K or 20-F filings found for {ticker}")
            return 0

        # Check if file already exists
        ticker_dir = os.path.join(self.output_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        accession_no = getattr(filing, "accession_no", None)
        if not accession_no:
            self.logger.warning(f"Missing accession number for {ticker}")
            return 0
        filename = f"{ticker}_{accession_no}.txt"
        filepath = os.path.join(ticker_dir, filename)
        metadata = self._capture_metadata(filing, ticker, form_type or "", filepath)

        if os.path.exists(filepath):
            self.logger.info(f"File already exists: {filename}")
            # Backfill the provenance sidecar if a prior run predates it.
            try:
                with open(filepath, encoding="utf-8") as f:
                    self._write_sidecar(filepath, f.read(), metadata)
            except OSError as e:
                self.logger.warning(f"Could not backfill sidecar for {filename}: {e}")
            return 1

        # Extract text (external API call - can fail)
        self.logger.info(f"Extracting {form_type} for {ticker}")
        try:
            text_attr = getattr(filing, "text", None)
            if not callable(text_attr):
                self.logger.error(f"Filing.text() not available for {ticker}")
                return 0
            raw_text = text_attr()
            if not isinstance(raw_text, str):
                self.logger.error(f"Filing.text() returned non-string for {ticker}")
                return 0
            text = self._normalize_text(raw_text)
        except Exception as e:  # external I/O/parsing errors
            self.logger.error(f"Failed to extract text for {ticker}: {e}")
            return 0

        if len(text) < 1000:
            self.logger.warning(
                f"Insufficient content for {ticker} ({len(text)} chars)"
            )
            return 0

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        self._write_sidecar(filepath, text, metadata)

        self.logger.info(f"Saved: {filename} ({len(text):,} chars)")
        return 1


def download_and_extract_10k(
    ticker: str,
    user_agent: str = DEFAULT_UA,
    output_dir: str = OUTPUT_BASE,
    forms: list[str] | None = None,
    year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 1,
) -> int:
    """Download 10-K filing for a ticker."""
    extractor = SECExtractor(user_agent, output_dir)
    return extractor.download_and_extract(ticker, forms, year, start_date, end_date)
