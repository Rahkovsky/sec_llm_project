"""Disclosure Change & Risk Signal Monitor orchestration (flagship)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..comparison.differ import FilingComparer
from ..core.types import Chunk
from .signals import RiskSignal, detect_signals

# Default forms the monitor inspects for risk signals.
DEFAULT_FORMS = ["10-K", "10-Q", "8-K"]
# SEC items most relevant to disclosure-change monitoring.
MONITOR_ITEMS = ["1A", "3", "7"]


@dataclass
class MonitorReport:
    ticker: str
    year_a: str
    year_b: str
    forms: list[str]
    compare_form: str
    risk_factor_changes: dict[str, Any] = field(default_factory=dict)
    new_litigation: list[str] = field(default_factory=list)
    signals_by_type: dict[str, list[RiskSignal]] = field(default_factory=dict)
    going_concern: bool = False
    xbrl_context: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": {"from": self.year_a, "to": self.year_b},
            "forms": self.forms,
            "compare_form": self.compare_form,
            "risk_factor_changes": self.risk_factor_changes,
            "new_litigation": self.new_litigation,
            "signals": {
                stype: [s.to_dict() for s in sigs]
                for stype, sigs in self.signals_by_type.items()
            },
            "going_concern": self.going_concern,
            "xbrl_context": self.xbrl_context,
            "summary": self.summary(),
            "notes": self.notes,
        }

    def summary(self) -> dict[str, Any]:
        rfc = self.risk_factor_changes or {}
        return {
            "new_risk_factors": len(rfc.get("added", [])),
            "removed_risk_factors": len(rfc.get("removed", [])),
            "reworded_risk_factors": len(rfc.get("changed", [])),
            "new_litigation_disclosures": len(self.new_litigation),
            "risk_signals": sum(len(v) for v in self.signals_by_type.values()),
            "going_concern": self.going_concern,
        }


class DisclosureMonitor:
    def __init__(self, pipeline: Any) -> None:
        # Duck-typed against SECIntelPipeline to avoid a circular import.
        self.pipeline = pipeline
        self.comparer = FilingComparer(pipeline.index)

    def _filing_chunks(self, ticker: str, fiscal_year: str,
                       forms: list[str]) -> list[Chunk]:
        out: list[Chunk] = []
        wanted = set(forms)
        for c in self.pipeline.index.all_chunks():
            m = c.metadata
            if m.ticker.upper() == ticker.upper() and m.fiscal_year == str(fiscal_year) \
                    and m.filing_type in wanted:
                out.append(c)
        return out

    def monitor(self, ticker: str, year_a: str, year_b: str, *,
                forms: list[str] | None = None, compare_form: str = "10-K",
                xbrl: bool = False) -> MonitorReport:
        forms = forms or DEFAULT_FORMS
        report = MonitorReport(
            ticker=ticker.upper(), year_a=str(year_a), year_b=str(year_b),
            forms=forms, compare_form=compare_form,
        )

        # 1) Risk-factor + litigation changes across the two annual filings.
        diff = self.comparer.compare_years(
            ticker, year_a, year_b, filing_type=compare_form, items=MONITOR_ITEMS
        ).to_dict()
        sections = {s["item_number"]: s for s in diff["sections"]}
        if "1A" in sections:
            report.risk_factor_changes = {
                "section_title": sections["1A"]["section_title"],
                "similarity": sections["1A"]["similarity"],
                "added": sections["1A"]["added"],
                "removed": sections["1A"]["removed"],
                "changed": sections["1A"]["changed"],
                "source": f"{ticker.upper()} {compare_form} {year_a} vs {year_b}",
            }
        if "3" in sections:
            report.new_litigation = sections["3"]["added"]
        if not sections:
            report.notes.append(
                f"No {compare_form} sections found for {ticker} {year_a}/{year_b}; "
                "ensure both filings are indexed."
            )

        # 2) Risk signals in the most recent period across requested forms.
        latest = self._filing_chunks(ticker, str(year_b), forms)
        if not latest:
            latest = self._filing_chunks(ticker, str(year_b), [compare_form])
        signals = detect_signals(latest)
        grouped: dict[str, list[RiskSignal]] = {}
        for sig in signals:
            grouped.setdefault(sig.signal_type, []).append(sig)
        report.signals_by_type = grouped
        report.going_concern = bool(grouped.get("going_concern"))

        # 3) Optional XBRL financial context (documented hook).
        if xbrl:
            report.xbrl_context = self._xbrl_context(ticker, str(year_b))
            if report.xbrl_context is None:
                report.notes.append(
                    "XBRL context unavailable (requires edgartools + network)."
                )

        return report

    @staticmethod
    def _xbrl_context(ticker: str, fiscal_year: str) -> dict[str, Any] | None:
        """Pull selected XBRL company facts via edgartools, if available.

        Returns None when the dependency or network is unavailable so the rest of
        the report still renders. Provenance: SEC XBRL company facts API.
        """
        try:
            from edgar import Company  # lazy, optional
        except Exception:
            return None
        try:
            company = Company(ticker)
            facts = company.get_facts()
            return {
                "ticker": ticker, "fiscal_year": fiscal_year,
                "source": "SEC XBRL company facts",
                "available": facts is not None,
            }
        except Exception:
            return None
