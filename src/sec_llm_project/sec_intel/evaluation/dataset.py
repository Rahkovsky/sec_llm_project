"""Curated SEC question set + a small synthetic corpus (plan items 9, 12).

The synthetic filings let the whole platform run, evaluate, and demo with no
network access or downloaded data. Real corpora are ingested the same way via
``IndexBuilder.build_from_dir``; this module just guarantees a deterministic,
self-contained baseline. Gold relevance is keyed on SEC item numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.types import FilingMetadata
from .runner import EvalCase

# (metadata, full filing text). Two companies, two fiscal years each.
SAMPLE_FILINGS: list[tuple[FilingMetadata, str]] = [
    (
        FilingMetadata(ticker="NOVA", company="Nova Devices Inc.", filing_type="10-K",
                       filing_date="2023-02-15", fiscal_year="2023", accession="NOVA-23-1"),
        """Item 1. Business
Nova Devices designs networking hardware and a subscription cloud platform sold to enterprises globally.

Item 1A. Risk Factors
We face intense competition from larger incumbents that could pressure our pricing and margins.
We rely on a small number of contract manufacturers in Asia, and supply chain disruptions or component shortages could materially harm our ability to ship products.
A cybersecurity breach of our cloud platform could expose customer data and subject us to liability.

Item 3. Legal Proceedings
We are defending a patent infringement suit brought by a competitor concerning our routing technology.

Item 7. Management's Discussion and Analysis of Financial Condition
Revenue grew 9% driven by subscription renewals. Our liquidity position is adequate, supported by operating cash flow and an undrawn revolving credit facility.

Item 9A. Controls and Procedures
Management concluded that our internal control over financial reporting was effective as of the fiscal year end, with no material weaknesses identified.
""",
    ),
    (
        FilingMetadata(ticker="NOVA", company="Nova Devices Inc.", filing_type="10-K",
                       filing_date="2024-02-14", fiscal_year="2024", accession="NOVA-24-1"),
        """Item 1. Business
Nova Devices designs networking hardware and a subscription cloud platform sold to enterprises globally, and in 2024 expanded into AI inference appliances.

Item 1A. Risk Factors
We face intense competition from larger incumbents that could pressure our pricing and margins.
We rely on a small number of contract manufacturers in Asia, and supply chain disruptions or component shortages could materially harm our ability to ship products.
A cybersecurity breach of our cloud platform could expose customer data and subject us to liability.
New in this fiscal year, evolving artificial intelligence regulations across multiple jurisdictions could increase our compliance costs and restrict certain product features.
Rising interest rates have increased the cost of our variable-rate borrowings and could constrain future liquidity.

Item 3. Legal Proceedings
The previously disclosed patent infringement suit was settled during the year on terms that were not material.

Item 7. Management's Discussion and Analysis of Financial Condition
Revenue grew 14% driven by AI appliance demand and subscription renewals. Liquidity tightened modestly as we funded inventory for the new product line, though we maintain access to our revolving credit facility.

Item 9A. Controls and Procedures
Management identified a material weakness related to controls over revenue recognition for multi-element arrangements and is implementing a remediation plan.
""",
    ),
    (
        FilingMetadata(ticker="ORCA", company="Orca Logistics Corp.", filing_type="10-K",
                       filing_date="2024-03-01", fiscal_year="2024", accession="ORCA-24-1"),
        """Item 1. Business
Orca Logistics operates a freight brokerage and last-mile delivery network across North America.

Item 1A. Risk Factors
Volatile fuel prices directly affect our cost of operations and could reduce profitability.
Driver shortages and labor costs could impair our ability to meet delivery commitments.
We are exposed to economic cycles; a downturn in consumer spending reduces shipment volumes.

Item 3. Legal Proceedings
We are party to a class action alleging misclassification of delivery drivers as independent contractors.

Item 7. Management's Discussion and Analysis of Financial Condition
Revenue declined 4% amid softening freight demand. We took on additional term debt to fund fleet expansion, and elevated leverage could pressure liquidity if cash flows weaken.

Item 13. Certain Relationships and Related Transactions
The company leases two distribution centers from an entity controlled by our chief executive officer on terms management believes are at market rates.
""",
    ),
]


# Curated questions with gold-relevant SEC items.
CURATED_CASES: list[EvalCase] = [
    EvalCase("q1", "What supply chain risks does Nova Devices face?", ["1A"], ticker="NOVA"),
    EvalCase("q2", "Summarize liquidity concerns for Orca Logistics.", ["7"], ticker="ORCA"),
    EvalCase("q3", "What litigation is Orca Logistics involved in?", ["3"], ticker="ORCA"),
    EvalCase("q4", "Does Nova Devices report any material weakness in internal controls?",
             ["9A"], ticker="NOVA"),
    EvalCase("q5", "What related-party transactions does Orca Logistics disclose?",
             ["13"], ticker="ORCA"),
    EvalCase("q6", "How do interest rates affect Nova Devices?", ["1A", "7"], ticker="NOVA"),
    EvalCase("q7", "What is the company's dividend payout ratio for 1998?",
             [], ticker="NOVA", answerable=False),
]


def index_sample_corpus(pipeline: object) -> object:
    """Index the built-in synthetic corpus into a pipeline (duck-typed)."""
    from ..pipeline import SECIntelPipeline

    assert isinstance(pipeline, SECIntelPipeline)
    all_chunks = []
    for meta, text in SAMPLE_FILINGS:
        all_chunks.extend(pipeline.builder.chunk_text(text, meta))
    return pipeline.index_chunks(all_chunks)


def load_cases_from_json(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=str(d["id"]), question=str(d["question"]),
            relevant_items=[str(x) for x in d.get("relevant_items", [])],
            ticker=d.get("ticker"), answerable=bool(d.get("answerable", True)),
            relevant_passages=[str(p) for p in d.get("relevant_passages", [])],
        )
        for d in data
    ]
