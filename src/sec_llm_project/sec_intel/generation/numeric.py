"""Numeric-equality checking for figures in grounded answers.

Financial answers live and die on numbers ("debt fell $2.5 billion", "margin was
14%"). Entailment/overlap judges are weak here: a wrong figure shares all the
surrounding vocabulary with the right one, so it slips through. This module
extracts figures from a claim and checks each one actually appears in the cited
evidence — a numeric counterpart to the verbatim-quote invariant.

Design notes
------------
* **Scale-aware, leniently.** "$1.5 billion" (claim) is matched against "1,500"
  in a table headed "(in millions)" by also trying x10^3/10^6/10^9 variants. This
  errs toward *not* flagging (a false numeric-hallucination flag would wrongly
  downgrade a correct answer), so implicit table scaling never causes a false hit.
* **Percent is its own class** — "14%" never matches a bare "14".
* **Tolerance** absorbs rounding ("1.58bn" vs "1.577bn", "14%" vs "14.0%").

Known limit (see FUTURE_WORK): values are compared by magnitude, not by the line
item they belong to — a figure present *somewhere* in the chunk counts as found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# number (with optional $ and thousands separators / decimals) + optional scale word
_FIG_RE = re.compile(
    r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*"
    r"(%|percent|percentage points?|basis points?|bps|trillion|billion|million|thousand|bn|mm|mn)?",
    re.IGNORECASE,
)
_SCALE = {
    "thousand": 1e3, "million": 1e6, "mm": 1e6, "mn": 1e6,
    "billion": 1e9, "bn": 1e9, "trillion": 1e12,
}
_PERCENT = {"%", "percent", "percentage point", "percentage points",
            "basis point", "basis points", "bps"}
_DEFAULT_TOL = 0.005          # 0.5% relative tolerance
_SCALE_VARIANTS = (1e3, 1e6, 1e9)


@dataclass(frozen=True)
class Figure:
    value: float
    is_percent: bool


def extract_figures(text: str) -> list[Figure]:
    """Pull numeric figures (value + percent flag, scale folded in) from text."""
    figures: list[Figure] = []
    for m in _FIG_RE.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower().strip()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit in _PERCENT:
            figures.append(Figure(value, True))
        else:
            figures.append(Figure(value * _SCALE.get(unit, 1.0), False))
    return figures


def _close(a: float, b: float, tol: float) -> bool:
    if a == 0 or b == 0:
        return a == b
    return abs(a - b) <= tol * max(abs(a), abs(b))


def _matches(claim: Figure, ev: Figure, tol: float) -> bool:
    if claim.is_percent != ev.is_percent:
        return False
    if _close(claim.value, ev.value, tol):
        return True
    if claim.is_percent:               # percentages are not rescaled
        return False
    # Tolerate implicit table scaling ("(in millions)") in either direction.
    return any(_close(claim.value, ev.value * s, tol) or _close(claim.value * s, ev.value, tol)
               for s in _SCALE_VARIANTS)


def figures_supported(claim: str, evidence: str, *, tol: float = _DEFAULT_TOL) -> bool:
    """True if every figure in ``claim`` appears in ``evidence`` (scale-tolerant).

    Vacuously true when the claim contains no figures — this guard only fires on
    claims that actually assert a number.
    """
    claim_figs = extract_figures(claim)
    if not claim_figs:
        return True
    ev_figs = extract_figures(evidence)
    if not ev_figs:
        return False
    return all(any(_matches(cf, ef, tol) for ef in ev_figs) for cf in claim_figs)
