"""Deterministic entity resolution: which company (ticker) is a question about?

Retrieval is scoped to one company by a ticker filter. Until now the ticker came
from the caller (CLI ``--ticker``, or the eval case itself -- an oracle). This
resolver derives it from the question text alone, with no model call:

1. **Company names** -- the issuer's registered name, normalized ("ALTRIA GROUP,
   INC." -> "altria group"), plus a distinctive short alias (the first word of
   the name when it is unique across issuers and not an English word, e.g.
   "altria", "colgate"). Matched on word boundaries, longest first.
2. **Tickers** -- upper-case tokens in the question that are known tickers
   (BRK.B style included). Tickers that are also ordinary words ("IT", "ON",
   "ALL", "KEY", "A", "L", ...) count only in an explicit ticker position:
   "(IT)", "$IT", "ticker IT", or at the very start/end of the question.

A question naming two companies resolves to both (comparison questions). One
naming none resolves to nothing; the caller then decides whether to search
everything, ask the user, or abstain -- the resolver never guesses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd",
    "limited", "plc", "llc", "lp", "nv", "sa", "ag", "se", "holdings", "holding", "group",
    "the", "de", "new", "trust", "international", "intl",
}
# Tickers that are also ordinary English / finance words: need an explicit ticker cue.
_AMBIGUOUS = {
    "A", "AI", "ALL", "AN", "ARE", "BE", "BIG", "CAN", "CPU", "DAY", "DE", "ED", "EL", "EQ",
    "ES", "ESG", "FOR", "GO", "HAS", "HE", "IP", "IR", "IT", "KEY", "LOW", "MA", "MAR", "NOW",
    "ON", "OR", "PM", "SO", "WELL", "AM", "US", "CEO", "CFO", "EPS", "ROI", "ROE",
}
_COMMON_WORDS = {
    "american", "general", "united", "first", "national", "public", "international",
    "global", "digital", "southern", "northern", "eastern", "western", "texas", "pacific",
    "best", "applied", "advanced", "american", "consolidated", "federal", "home", "health",
    "energy", "capital", "financial", "bank", "realty", "medical", "air", "royal", "regions",
    "ball", "target", "progressive", "corning", "visa", "oracle", "intuitive", "edison",
    "public", "republic", "universal", "waste", "gap", "lowe", "dollar", "match", "fox",
    "news", "booking", "airbnb", "lennar", "salesforce", "workday", "align", "insulet",
    "steel", "state", "street", "the", "old", "dominion", "duke", "brown", "baker", "cboe",
    "constellation", "simon", "church", "masco", "carrier", "ross", "tractor", "arch",
}


def _norm(text: str) -> str:
    text = text.lower().replace("&", " and ").replace("’", "'")
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"/[a-z]{2,3}/", " ", text)  # EDGAR state tags: "/DE/"
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _name_key(company: str) -> str:
    words = [w for w in _norm(company).split() if w not in _SUFFIXES]
    return " ".join(words)


# "&" excluded on both sides: "R&D", "M&A", "S&P" are not tickers D / A / P.
_TICKER_RX = re.compile(r"(?<![A-Za-z0-9.&])\$?([A-Z]{1,5}(?:\.[A-Z])?)(?![A-Za-z0-9&])")


def _phrase_rx(phrase: str) -> re.Pattern[str]:
    """Normalized name -> case-insensitive regex over the raw question text."""
    parts = [r"(?:and|&)" if w == "and" else re.escape(w) for w in phrase.split()]
    body = r"[\s\-/.,&']*".join(parts)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?P<poss>['’]s)?(?![A-Za-z0-9])",
                      re.IGNORECASE)


@dataclass
class Resolution:
    tickers: list[str]
    method: str  # name | ticker | name+ticker | none
    evidence: dict[str, str] = field(default_factory=dict)  # ticker -> matched text

    @property
    def resolved(self) -> bool:
        return bool(self.tickers)


class EntityResolver:
    def __init__(self, companies: dict[str, str]) -> None:
        """``companies``: ticker -> registered company name."""
        self.tickers = {t.upper() for t in companies}
        self.names = {t.upper(): n for t, n in companies.items()}  # registered names
        self._names: list[tuple[str, str]] = []  # (normalized phrase, ticker)
        by_name: dict[str, set[str]] = {}
        for t, name in companies.items():
            key = _name_key(name)
            if key:
                by_name.setdefault(key, set()).add(t.upper())
        # share classes (GOOG/GOOGL, BRK.A/B) map one name to several tickers: keep all
        firsts = Counter(k.split()[0] for k in by_name)
        for key, ts in by_name.items():
            for t in sorted(ts):
                self._names.append((key, t))
                first = key.split()[0]
                if (" " in key and firsts[first] == 1 and len(first) >= 5
                        and first not in _COMMON_WORDS):
                    self._names.append((first, t))
        self._names.sort(key=lambda p: -len(p[0]))  # longest phrase wins
        self._name_rx = [(ph, t, _phrase_rx(ph)) for ph, t in self._names]
        self._share_class = {t: ts for ts in by_name.values() for t in ts if len(ts) > 1}

    @classmethod
    def from_input_dirs(cls, dirs: Iterable[str | Path]) -> EntityResolver:
        """Build from the downloader's per-filing ``.json`` sidecars."""
        names: dict[str, Counter] = {}
        for d in dirs:
            for f in Path(d).glob("*/*.json"):
                try:
                    m = json.loads(f.read_text(encoding="utf-8")).get("metadata", {})
                except (OSError, json.JSONDecodeError):
                    continue
                t, c = m.get("ticker"), m.get("company")
                if t and c and c != "UNKNOWN":
                    names.setdefault(t, Counter())[c] += 1
        return cls({t: c.most_common(1)[0][0] for t, c in names.items()})

    def resolve(self, question: str) -> Resolution:
        """Tickers ranked: the question's main company first.

        Rank: named by both name and ticker ("Fastenal (FAST)") > possessive
        ("AVY's") > earliest mention. Peers named in comparisons follow.
        """
        hits: dict[str, dict] = {}  # ticker -> {pos, how, text, poss}
        taken: list[tuple[int, int]] = []
        for phrase, t, rx in self._name_rx:
            for m in rx.finditer(question):
                if not (m.group(0)[0].isupper() or m.group(0)[0].isdigit()):
                    continue  # "union coverage", "threat intel": not a proper noun
                if any(a < m.end() and m.start() < b for a, b in taken):
                    continue  # inside a longer name already matched
                taken.append((m.start(), m.end()))
                h = hits.setdefault(t, {"pos": m.start(), "how": set(), "text": m.group(0).strip(),
                                        "poss": False})
                h["how"].add("name")
                h["poss"] |= bool(m.group("poss"))
        for m in _TICKER_RX.finditer(question):
            tok = m.group(1)
            if tok not in self.tickers:
                continue
            if ((tok in _AMBIGUOUS or len(tok) == 1)  # 1-letter tickers: A, C, D, F, T, V, ...
                    and not self._ticker_cue(question, m.start(), m.end())):
                continue
            h = hits.setdefault(tok, {"pos": m.start(), "how": set(), "text": tok, "poss": False})
            h["how"].add("ticker")
            h["pos"] = min(h["pos"], m.start())
            h["poss"] |= bool(re.match(r"['’]s\b", question[m.end():]))
        # one company with share classes (GOOG/GOOGL) found only by name: keep one
        for t in list(hits):
            sib = (self._share_class.get(t, set()) - {t}) & set(hits)
            if sib and hits[t]["how"] == {"name"}:
                del hits[t]
        order = sorted(hits, key=lambda t: (-len(hits[t]["how"]), not hits[t]["poss"],
                                            hits[t]["pos"]))
        how = sorted({x for t in order for x in hits[t]["how"]})
        return Resolution(tickers=order, method="+".join(how) or "none",
                          evidence={t: hits[t]["text"] for t in order})

    @staticmethod
    def _ticker_cue(q: str, start: int, end: int) -> bool:
        before, after = q[:start].rstrip(), q[end:].lstrip()
        if before.endswith(("(", "$")) and after.startswith((")", ",", "'", "’")):
            return True
        if q[start] == "$" or re.search(r"(?i)ticker\s*[:\-]?\s*$", before):
            return True
        if not before or not after.strip(" .?!"):  # first or last token
            return True
        return bool(re.search(r"(?i)^('s|’s)\b", after))  # possessive: "IT's"

    def mask(self, question: str, tickers: Iterable[str] = ()) -> str:
        """Company-blind question: every mention of these companies (and of any the
        resolver finds) -> "the company". Used to keep the company out of LLM query
        rewriting, where a named company makes the model refuse or recall from memory."""
        q = question
        tickers = set(tickers) | set(self.resolve(q).tickers)
        for t in tickers:
            q = _mask_one(q, t, self.names.get(t), self)
        return q


_GENERIC_NAME_WORDS = {
    "corp", "corporation", "incorporated", "company", "companies", "group", "holding",
    "holdings", "international", "intl", "limited", "trust", "energy", "financial", "services",
    "technologies", "technology", "systems", "industries", "global", "national", "american",
    "united", "general", "first", "capital", "bank", "health", "brands", "resources", "the",
    "and", "new", "partners", "properties", "realty", "communications", "solutions",
}


def _mask_one(q: str, ticker: str, full: str | None, resolver: EntityResolver) -> str:
    if full and _name_key(full):
        q = _phrase_rx(_name_key(full)).sub("the company", q)
    for sp in sorted(set(resolver.resolve(q).evidence.values()) | {ticker}, key=len, reverse=True):
        flags = 0 if sp.isupper() else re.IGNORECASE
        q = re.sub(r"(?<![A-Za-z0-9])\$?" + re.escape(sp) + r"(?![A-Za-z0-9])", "the company",
                   q, flags=flags)
    # leftover distinctive words of the registered name ("C.H. Robinson", "Fifth Third")
    for w in re.findall(r"[A-Za-z]{4,}", full or ""):
        if w.lower() not in _GENERIC_NAME_WORDS:
            q = re.sub(r"(?<![A-Za-z])" + re.escape(w) + r"(?![A-Za-z])", "the company", q,
                       flags=re.IGNORECASE)
    q = re.sub(r"(?:\b[A-Z]\.\s*)+the company", "the company", q)  # initials: C.H., J.
    q = re.sub(r"the company(?:\s+(?:[A-Z]\.\s*)*the company)+", "the company", q)
    q = re.sub(r"(the company)(?:\s+(?:Inc|Incorporated|Corp|Corporation|Co|Company|Companies|"
               r"Group|Holdings?|Ltd|plc|N\.?V|S\.?A|& Co)\.?)+", r"\1", q, flags=re.IGNORECASE)
    q = re.sub(r"\s*\((?:ticker:?\s*)?the company\)", "", q)
    q = re.sub(r"\bthe company(?:\s*,\s*the company)+", "the company", q)
    return re.sub(r"\s{2,}", " ", q).strip()
