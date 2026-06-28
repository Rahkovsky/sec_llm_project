"""Multi-source SEC ingestion (plan item 7).

Extends ingestion beyond 10-K to 10-Q, 8-K, and DEF 14A by reusing the existing
``edgartools``-based downloader. Each form type is written to its own directory
so :class:`IndexBuilder` can infer ``filing_type`` for provenance. XBRL company
facts can be layered on top via ``edgartools`` for numeric provenance; that path
is documented but kept optional to avoid a hard dependency in the core package.
"""

from __future__ import annotations

from .rate_limit import SEC_MAX_RPS, RateLimiter, build_user_agent
from .sec_download import CHECKPOINT_FILENAME, FORM_DIRS, FetchCheckpoint, download_filings
from .sp500 import get_sp500_tickers

__all__ = [
    "CHECKPOINT_FILENAME",
    "FORM_DIRS",
    "SEC_MAX_RPS",
    "FetchCheckpoint",
    "RateLimiter",
    "build_user_agent",
    "download_filings",
    "get_sp500_tickers",
]
