"""Shared, dependency-free data types used across the platform.

These dataclasses are the lingua franca between the chunking, indexing,
retrieval, extraction, generation, and comparison layers. They depend only on
the standard library so the whole core is importable and testable without any
ML/vector-store backends installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FilingMetadata:
    """Provenance for a single SEC filing.

    Every chunk derived from a filing carries a copy of this metadata so that
    answers can be fully attributed back to the source document (plan items 4,
    7, 8, 10).
    """

    ticker: str = "UNKNOWN"
    company: str = "UNKNOWN"
    cik: str = "UNKNOWN"
    filing_type: str = "UNKNOWN"  # 10-K, 10-Q, 8-K, DEF 14A, ...
    filing_date: str = "UNKNOWN"  # ISO date or year
    fiscal_year: str = "UNKNOWN"
    accession: str = "UNKNOWN"
    source_url: str = ""
    source_path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilingMetadata:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: str(v) for k, v in data.items() if k in known})


@dataclass
class Chunk:
    """A retrievable unit of text with full provenance.

    ``section_title`` / ``item_number`` capture SEC-aware structure so that
    citations can name the section a passage came from (plan item 4).
    """

    chunk_id: str
    text: str
    metadata: FilingMetadata
    section_title: str = "UNKNOWN"
    item_number: str = "UNKNOWN"
    char_start: int = 0
    char_end: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        # Cheap, deterministic proxy (~4 chars/token) used for budgeting.
        return max(1, len(self.text) // 4)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata"] = self.metadata.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        meta = data.get("metadata", {})
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data.get("text", "")),
            metadata=FilingMetadata.from_dict(meta) if isinstance(meta, dict) else FilingMetadata(),
            section_title=str(data.get("section_title", "UNKNOWN")),
            item_number=str(data.get("item_number", "UNKNOWN")),
            char_start=int(data.get("char_start", 0)),
            char_end=int(data.get("char_end", 0)),
            extra=dict(data.get("extra", {})),
        )


def make_chunk_id(metadata: FilingMetadata, item_number: str, ordinal: int) -> str:
    """Deterministic, collision-resistant chunk identifier.

    Stable across rebuilds for the same filing/section/ordinal, which keeps the
    index reproducible (plan item 11) and lets evaluation reference fixed ids.
    The basis includes a per-filing discriminator (source filename when present,
    else accession + fiscal year) so two filings of the same company cannot
    collide even when accession metadata is missing.
    """
    filing_key = metadata.source_path.rsplit("/", 1)[-1] or metadata.accession
    basis = "|".join(
        [metadata.ticker, filing_key, metadata.fiscal_year, metadata.filing_type,
         item_number, str(ordinal)]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"{metadata.ticker}:{item_number}:{ordinal}:{digest}"


@dataclass
class RetrievalResult:
    """A scored chunk returned from retrieval, with score breakdown for transparency."""

    chunk: Chunk
    score: float
    rank: int = 0
    components: dict[str, float] = field(default_factory=dict)  # bm25/dense/rerank/rrf

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "rank": self.rank,
            "components": self.components,
        }


@dataclass
class Citation:
    """A grounded reference attached to a generated answer (plan item 8)."""

    chunk_id: str
    ticker: str
    company: str
    filing_type: str
    filing_date: str
    section_title: str
    item_number: str
    source_url: str
    score: float
    quote: str
    # Populated by the citation verifier (empty when verification is off).
    verdict: str = ""          # SUPPORTED | PARTIAL | UNSUPPORTED | CONTRADICTED
    support_quote: str = ""    # verbatim span the judge matched to the claim

    @classmethod
    def from_result(cls, result: RetrievalResult, quote_chars: int = 240) -> Citation:
        return cls.from_chunk(result.chunk, score=result.score, quote_chars=quote_chars)

    @classmethod
    def from_chunk(cls, c: Chunk, *, score: float = 1.0, quote: str | None = None,
                   quote_chars: int = 240) -> Citation:
        m = c.metadata
        text = quote if quote is not None else c.text
        return cls(
            chunk_id=c.chunk_id,
            ticker=m.ticker,
            company=m.company,
            filing_type=m.filing_type,
            filing_date=m.filing_date,
            section_title=c.section_title,
            item_number=c.item_number,
            source_url=m.source_url,
            score=round(score, 4),
            quote=" ".join(text.split())[:quote_chars],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """A grounded answer with citations, confidence, and abstention support."""

    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstained: bool = False
    abstain_reason: str = ""
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "model": self.model,
            "extra": self.extra,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# Verdict vocabulary for claim-level citation verification.
SUPPORTED = "SUPPORTED"
PARTIAL = "PARTIAL"
UNSUPPORTED = "UNSUPPORTED"
CONTRADICTED = "CONTRADICTED"


@dataclass
class ClaimVerdict:
    """One atomic claim from an answer, judged against the cited evidence."""

    claim: str
    verdict: str  # SUPPORTED | PARTIAL | UNSUPPORTED | CONTRADICTED
    citation: int | None = None  # 1-based index into the evidence shown to the judge
    quote: str = ""              # verbatim supporting span from the cited evidence
    valid_quote: bool = False    # True iff `quote` is actually a substring of the cited chunk
    numeric_ok: bool = True      # False iff a figure in the claim is absent from the cited chunk

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    """Aggregate result of verifying an answer's claims (plan: responsible AI)."""

    claims: list[ClaimVerdict] = field(default_factory=list)
    judge_model: str = ""
    repaired: bool = False

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == SUPPORTED)

    @property
    def contradicted(self) -> int:
        return sum(1 for c in self.claims if c.verdict == CONTRADICTED)

    @property
    def groundedness(self) -> float:
        """Fraction of claims supported; PARTIAL counts as half. 0.0 when empty."""
        if not self.claims:
            return 0.0
        score = sum(
            1.0 if c.verdict == SUPPORTED else 0.5 if c.verdict == PARTIAL else 0.0
            for c in self.claims
        )
        return score / len(self.claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groundedness": round(self.groundedness, 4),
            "supported": self.supported,
            "contradicted": self.contradicted,
            "total": self.total,
            "repaired": self.repaired,
            "judge_model": self.judge_model,
            "claims": [c.to_dict() for c in self.claims],
        }
