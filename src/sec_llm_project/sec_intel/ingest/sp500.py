"""S&P 500 constituent ticker list — bundled snapshot + live Wikipedia refresh.

The bundled list reflects the index as of approximately Q1 2025 (~500 symbols;
some companies have two share-class entries, e.g. GOOG/GOOGL).  It is a
best-effort snapshot and will drift as the index reconstitutes.

For the current official list call ``get_sp500_tickers(live=True)``, which
fetches the Wikipedia table using only stdlib ``urllib`` / ``html.parser`` and
requires a network connection.
"""

from __future__ import annotations

# ~500 symbols as of Q1 2025.  Use get_sp500_tickers(live=True) for the
# current list.  BRK.B and BF.B use the dot notation edgartools accepts.
_BUNDLED: tuple[str, ...] = (
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AFRM", "AIG", "AIZ", "AJG",
    "AKAM", "ALB", "ALGN", "ALL", "ALLE", "ALLY", "AMAT", "AMCR", "AMD", "AME",
    "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD",
    "APH", "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO",
    "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF.B", "BG",
    "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BMY", "BR", "BRK.B", "BRO",
    "BSX", "BWA", "BXP",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL",
    "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHTR", "CHRW", "CI",
    "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM",
    "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX",
    "D", "DAL", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLTR", "DLR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "ELV", "EMN", "EMR",
    "ENPH", "EOG", "EPAM", "EQIX", "EQR", "ES", "ESS", "ETN", "ETR", "ETSY",
    "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS",
    "FISV", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTV",
    "GD", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLPI", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HOLX", "HON",
    "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU",
    "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ",
    "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM",
    "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KO",
    "KR", "KVUE",
    "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC",
    "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
    "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRO", "MRNA", "MS", "MSCI",
    "MSFT", "MSI", "MTCH", "MTB", "MTD", "MU",
    "NCLH", "NDAQ", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG",
    "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA",
    "O", "ODFL", "OGN", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
    "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PODD", "POOL", "PPG",
    "PPL", "PRU", "PSA", "PSX", "PTC", "PWR",
    "QCOM", "QRVO",
    "RCL", "RE", "REG", "REGN", "RF", "RHI", "RJF", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG", "RTX", "RVTY",
    "SBAC", "SBUX", "SCHW", "SEE", "SHW", "SJM", "SLB", "SNA", "SNPS", "SO",
    "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK",
    "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
    "TJX", "TMO", "TMUS", "TOL", "TPR", "TRGP", "TRMB", "TRV", "TSCO", "TSLA",
    "TSN", "TT", "TTWO", "TXN",
    "UAL", "UDR", "UHS", "ULTA", "UNH", "UNM", "UNP", "UPS", "URI", "USB",
    "V", "VFC", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX", "VTRS", "VZ",
    "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM",
    "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XYL",
    "YUM",
    "ZBH", "ZBRA", "ZION", "ZTS",
)


def _fetch_live() -> list[str]:
    """Scrape the current S&P 500 table from Wikipedia (stdlib only)."""
    import urllib.request
    from html.parser import HTMLParser

    class _ConstituentParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._in_table = False
            self._in_first_td = False
            self._cell_index = 0
            self._current: list[str] = []
            self.tickers: list[str] = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag == "table" and dict(attrs).get("id") == "constituents":
                self._in_table = True
            if self._in_table:
                if tag == "tr":
                    self._cell_index = 0
                if tag == "td":
                    self._cell_index += 1
                    if self._cell_index == 1:
                        self._in_first_td = True
                        self._current = []

        def handle_endtag(self, tag: str) -> None:
            if tag == "table":
                self._in_table = False
            if self._in_table and tag == "td" and self._in_first_td:
                self._in_first_td = False
                ticker = "".join(self._current).strip()
                if ticker:
                    self.tickers.append(ticker)

        def handle_data(self, data: str) -> None:
            if self._in_first_td:
                self._current.append(data)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "sec-intel/1.0 (research tool)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    parser = _ConstituentParser()
    parser.feed(html)
    return sorted(set(parser.tickers))


def get_sp500_tickers(*, live: bool = False) -> list[str]:
    """Return S&P 500 ticker symbols.

    Parameters
    ----------
    live:
        When ``True``, fetch the current list from Wikipedia (requires
        network).  When ``False`` (default), return the bundled snapshot
        from Q1 2025 — suitable for offline use and CI.
    """
    if live:
        return _fetch_live()
    return list(_BUNDLED)
