"""Named periods in a question ("fiscal 2023", "FY24", "in 2024") -> fiscal-year filter.

A question about year Y is answered by the filings of fiscal year Y and by the
next year's 10-K, which restates Y as the comparative period. ``period_filter``
returns that set of fiscal years, restricted to the years the ticker actually
has; when none of them exists (e.g. "fiscal 2028") it returns None and retrieval
stays unfiltered, so the generator sees the nearest filings and can say the
period is not covered.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_YEAR_RE = re.compile(r"(?<![\d$.,])((?:19|20)\d\d)(?![\d,%]|\.\d)")
_FY2_RE = re.compile(r"\b(?:FY|fiscal(?:\s+year)?)\s*'?(\d\d)\b", re.IGNORECASE)


def named_years(question: str) -> list[int]:
    """Calendar/fiscal years named in the question (1990-2039), in order, unique."""
    out: list[int] = []
    for m in _YEAR_RE.finditer(question):
        y = int(m.group(1))
        if 1990 <= y <= 2039 and y not in out:
            out.append(y)
    for m in _FY2_RE.finditer(question):
        y = 2000 + int(m.group(1))
        if y <= 2039 and y not in out:
            out.append(y)
    return out


def period_filter(question: str, available: Iterable[str]) -> list[str] | None:
    """Fiscal years to keep for ``question`` (named years and the year after), or None."""
    years = named_years(question)
    if not years:
        return None
    have = {str(a) for a in available}
    want = sorted({str(y) for y in years} | {str(y + 1) for y in years})
    keep = [y for y in want if y in have]
    return keep or None
