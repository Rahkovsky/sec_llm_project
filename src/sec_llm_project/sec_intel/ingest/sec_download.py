"""Download SEC filings of several form types for a set of tickers (plan item 7).

Official EDGAR sources only. Requests are rate-limited under SEC's fair-access
policy (<= 10 req/s) and sent with a descriptive User-Agent. Orchestrates the
project's existing ``SECExtractor`` (edgartools), imported lazily so importing
the platform never requires network/SEC packages.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterable

from .rate_limit import DEFAULT_RPS, RateLimiter, build_user_agent

CHECKPOINT_FILENAME = ".fetch_checkpoint.json"


class FetchCheckpoint:
    """Atomic JSON checkpoint for resumable bulk EDGAR downloads.

    Records the result of every (ticker, form, year) attempt:
      "ok"      — filing downloaded (file exists on disk)
      "missing" — EDGAR had no filing; skipped on resume

    Items absent from the file have not been attempted and will be tried.
    Use ``clear()`` (or ``sec-intel fetch --reset``) to start fresh.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        try:
            with open(path) as f:
                self._data: dict[str, str] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    @staticmethod
    def key(ticker: str, form: str, yr: int | None) -> str:
        return f"{ticker}|{form}|{yr or 'latest'}"

    def is_done(self, key: str) -> bool:
        """True for both "ok" and "missing" — skip on resume."""
        return key in self._data

    def mark(self, key: str, state: str) -> None:
        self._data[key] = state
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f)
        os.replace(tmp, self._path)  # atomic — crash-safe

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for v in self._data.values():
            c[v] = c.get(v, 0) + 1
        return c

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self._path)
        self._data = {}

# Where each form type lands; IndexBuilder maps these back to filing_type.
FORM_DIRS: dict[str, str] = {
    "10-K": "data/input/10-K",
    "10-Q": "data/input/10-Q",
    "8-K": "data/input/8-K",
    "DEF 14A": "data/input/DEF14A",
}


def _resolve_user_agent(user_agent: str | None) -> str:
    if user_agent:
        return user_agent
    # Build a compliant UA from SEC_USER_NAME / SEC_USER_EMAIL if available.
    try:
        from ...utils.env_config import get_sec_user_info

        name, email = get_sec_user_info()
    except Exception:
        name, email = "Anonymous", "anonymous@example.com"
    return build_user_agent(name, email)


def download_filings(tickers: Iterable[str], forms: list[str] | None = None, *,
                     year: int | None = None, years: Iterable[int] | None = None,
                     base_dir: str = "data/input",
                     user_agent: str | None = None,
                     rate_per_sec: float = DEFAULT_RPS,
                     reset: bool = False) -> dict[str, int]:
    """Download the requested form types for each ticker across one or more years.

    The flagship Disclosure Monitor and the comparison engine both diff filings
    *year over year*, so pass ``years=[2023, 2024]`` (or a wider range) to build a
    corpus that supports them. ``year`` is kept as a single-year convenience and
    is merged with ``years``; if neither is given, the latest filing is fetched.

    Forms land under ``data/input/<FORM>/<TICKER>/`` so the indexer can recover
    the filing type from the path. Returns a mapping of form -> count downloaded.

    A checkpoint at ``{base_dir}/.fetch_checkpoint.json`` records each (ticker,
    form, year) after its first attempt so interrupted runs resume without
    re-querying EDGAR.  Pass ``reset=True`` to clear it and start fresh.
    """
    from ...download.core import SECExtractor  # lazy: pulls in edgartools

    ua = _resolve_user_agent(user_agent)
    limiter = RateLimiter(rate_per_sec=rate_per_sec)
    forms = forms or ["10-K"]
    tickers = list(tickers)

    # Normalise the requested years into a sorted, de-duplicated list.
    year_list: list[int | None] = sorted({*(years or []), *([year] if year else [])})
    if not year_list:
        year_list = [None]  # latest filing only

    cp_path = os.path.join(base_dir, CHECKPOINT_FILENAME)
    checkpoint = FetchCheckpoint(cp_path)
    if reset:
        checkpoint.clear()
        print(f"Checkpoint cleared: {cp_path}")
    else:
        prior = checkpoint.counts()
        if prior:
            n_ok = prior.get("ok", 0)
            n_miss = prior.get("missing", 0)
            print(f"Resuming — checkpoint: {n_ok} done, {n_miss} confirmed-missing "
                  f"(pass --reset to start fresh)  [{cp_path}]")

    results: dict[str, int] = dict.fromkeys(forms, 0)
    n_skipped = 0

    for form in forms:
        out_dir = FORM_DIRS.get(form, os.path.join(base_dir, form.replace(" ", "")))
        os.makedirs(out_dir, exist_ok=True)
        extractor = SECExtractor(user_agent=ua, output_dir=out_dir)
        for ticker in tickers:
            for yr in year_list:
                key = FetchCheckpoint.key(ticker, form, yr)
                if checkpoint.is_done(key):
                    n_skipped += 1
                    continue

                limiter.acquire()  # respect SEC fair-access before each network call
                count = extractor.download_and_extract(ticker, forms=[form], year=yr)

                # Checkpoint persists immediately — crash-safe via atomic rename.
                checkpoint.mark(key, "ok" if count > 0 else "missing")
                results[form] += count

    if n_skipped:
        print(f"Skipped {n_skipped} already-processed items (checkpoint)")
    return results
