"""Disclosure Change & Risk Signal Monitor (flagship workflow).

Given a ticker, two fiscal years, and one or more form types, surfaces:
  * changed risk-factor language,
  * new legal/litigation disclosures,
  * liquidity / going-concern signals,
  * optional XBRL financial context,
all citation-grounded against official SEC filings.
"""

from __future__ import annotations

from .monitor import DisclosureMonitor, MonitorReport
from .signals import SIGNAL_LEXICON, RiskSignal, detect_signals

__all__ = [
    "SIGNAL_LEXICON",
    "DisclosureMonitor",
    "MonitorReport",
    "RiskSignal",
    "detect_signals",
]
