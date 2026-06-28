"""Enforcement-case benchmark — evaluation, not training.

Public SEC enforcement actions (e.g. Accounting and Auditing Enforcement
Releases) describe categories of disclosure failure: omitted going-concern
warnings, revenue-recognition material weaknesses, undisclosed related-party
self-dealing, liquidity/covenant breaches, and so on. We use these *categories*
as a benchmark to ask a narrow, defensible question:

    "Given filings, does the system retrieve the relevant evidence and surface
     the known risk signals an analyst would expect to see?"

This evaluates retrieval and organization. It is NOT a predictive model of
wrongdoing, and the benchmark makes no claim that any specific company committed
a violation. Each case points at a public SEC resource so reviewers can trace
the signal taxonomy back to official sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..monitor.signals import detect_signals
from . import metrics

# Public, stable SEC enforcement resources (taxonomy provenance only).
AAER_INDEX_URL = "https://www.sec.gov/divisions/enforce/friactions.htm"
LITIGATION_RELEASES_URL = "https://www.sec.gov/litigation/litreleases"


@dataclass
class EnforcementCase:
    id: str
    description: str            # category of disclosure issue under test
    expected_signals: list[str]  # signal types the system should surface
    expected_items: list[str]    # SEC items where the evidence should live
    ticker: str | None = None  # entity in the indexed corpus to inspect
    source_url: str = AAER_INDEX_URL  # public SEC resource for the taxonomy

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "description": self.description,
            "expected_signals": self.expected_signals,
            "expected_items": self.expected_items,
            "ticker": self.ticker, "source_url": self.source_url,
        }


# A small benchmark wired to the built-in synthetic corpus so it runs offline.
# Replace/extend with real entities + AAER URLs by indexing real filings and
# loading cases from JSON (see load_enforcement_cases).
ENFORCEMENT_CASES: list[EnforcementCase] = [
    EnforcementCase(
        id="rev-rec-material-weakness",
        description="Revenue-recognition internal-control material weakness "
                    "(common AAER theme); system should surface the weakness "
                    "and any liquidity strain.",
        expected_signals=["material_weakness", "liquidity"],
        expected_items=["9A", "7", "1A"],
        ticker="NOVA",
    ),
    EnforcementCase(
        id="worker-classification-litigation",
        description="Disclosed litigation exposure (e.g. worker misclassification "
                    "class action) plus leverage/liquidity pressure.",
        expected_signals=["litigation", "liquidity"],
        expected_items=["3", "7"],
        ticker="ORCA",
    ),
    EnforcementCase(
        id="related-party-self-dealing",
        description="Undisclosed/under-disclosed related-party transactions "
                    "(self-dealing) — evidence should retrieve from Item 13.",
        expected_signals=["litigation"],   # proxy signal; full case adds related_party
        expected_items=["13"],
        ticker="ORCA",
    ),
]


@dataclass
class EnforcementReport:
    n_cases: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_case: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_cases": self.n_cases, "metrics": self.metrics, "per_case": self.per_case}

    def render(self) -> str:
        lines = [f"Enforcement-case benchmark ({self.n_cases} cases) — evaluation only", "-" * 50]
        lines.extend(f"  {key:26s}: {self.metrics[key]:.4f}" for key in sorted(self.metrics))
        return "\n".join(lines)


def run_enforcement_benchmark(pipeline: Any, cases: list[EnforcementCase] | None = None,
                              *, k: int = 8) -> EnforcementReport:
    """Measure signal coverage and evidence retrieval for each case."""
    cases = cases or ENFORCEMENT_CASES
    signal_recalls: list[float] = []
    evidence_recalls: list[float] = []
    per_case: list[dict[str, Any]] = []

    for case in cases:
        chunks = [
            c for c in pipeline.index.all_chunks()
            if not case.ticker or c.metadata.ticker.upper() == case.ticker.upper()
        ]
        found_signals = {s.signal_type for s in detect_signals(chunks)}
        signal_recall = metrics.recall_at_k(
            list(found_signals), case.expected_signals, len(found_signals) or 1
        )
        signal_recalls.append(signal_recall)

        filters = {"ticker": case.ticker.upper()} if case.ticker else None
        results = pipeline.retriever.retrieve(case.description, top_k=k, filters=filters)
        retrieved_items: list[str] = []
        for r in results:
            if r.chunk.item_number not in retrieved_items:
                retrieved_items.append(r.chunk.item_number)
        evidence_recall = metrics.recall_at_k(retrieved_items, case.expected_items, k)
        evidence_recalls.append(evidence_recall)

        per_case.append({
            "id": case.id, "source_url": case.source_url,
            "expected_signals": case.expected_signals,
            "found_signals": sorted(found_signals),
            "signal_recall": round(signal_recall, 4),
            "evidence_recall": round(evidence_recall, 4),
            "retrieved_items": retrieved_items[:k],
        })

    n = len(cases)
    agg = {
        "signal_recall": sum(signal_recalls) / n if n else 0.0,
        "evidence_retrieval_recall": sum(evidence_recalls) / n if n else 0.0,
    }
    return EnforcementReport(n_cases=n, metrics=agg, per_case=per_case)


def load_enforcement_cases(path: str | Path) -> list[EnforcementCase]:
    """Load real enforcement cases (with AAER/litigation-release URLs) from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EnforcementCase(
            id=str(d["id"]), description=str(d["description"]),
            expected_signals=[str(x) for x in d.get("expected_signals", [])],
            expected_items=[str(x) for x in d.get("expected_items", [])],
            ticker=d.get("ticker"),
            source_url=str(d.get("source_url", AAER_INDEX_URL)),
        )
        for d in data
    ]
