"""Deterministic risk-signal detection over filing chunks.

A transparent, auditable lexicon of regulatory risk phrases. Each detected
signal is grounded in the exact chunk (and quote) it came from, so nothing is
asserted without a citation. This is detection of *disclosed* language, not a
predictive model — consistent with "organize known risk signals", not "train a
detector".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import Chunk, Citation

# signal_type -> indicative phrases (lowercase). Ordered by regulatory severity.
SIGNAL_LEXICON: dict[str, list[str]] = {
    "going_concern": [
        "going concern", "substantial doubt", "ability to continue as a going concern",
    ],
    "liquidity": [
        "insufficient liquidity", "liquidity constraints", "may not have sufficient",
        "default on our", "covenant", "breach of covenant", "additional financing",
        "negative working capital", "constrain future liquidity",
    ],
    "material_weakness": [
        "material weakness", "significant deficiency", "were not effective",
        "was not effective", "remediation plan",
    ],
    "litigation": [
        "class action", "lawsuit", "litigation", "investigation", "subpoena",
        "enforcement action", "alleging", "infringement",
    ],
    "impairment": [
        "impairment", "goodwill impairment", "write-down", "write down",
    ],
    "restatement": [
        "restate", "restatement", "material misstatement",
    ],
}


@dataclass
class RiskSignal:
    signal_type: str
    phrase: str
    citation: Citation
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "signal_type": self.signal_type,
            "phrase": self.phrase,
            "snippet": self.snippet,
            "citation": self.citation.to_dict(),
        }


def _snippet_around(text: str, phrase: str, width: int = 160) -> str:
    idx = text.lower().find(phrase)
    if idx < 0:
        return " ".join(text.split())[:width]
    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(phrase) + width // 2)
    return " ".join(text[start:end].split())


def detect_signals(chunks: list[Chunk], *,
                   signal_types: list[str] | None = None) -> list[RiskSignal]:
    """Scan chunks for risk phrases; one signal per (chunk, phrase) match."""
    types = signal_types or list(SIGNAL_LEXICON)
    signals: list[RiskSignal] = []
    for chunk in chunks:
        low = chunk.text.lower()
        for stype in types:
            for phrase in SIGNAL_LEXICON.get(stype, []):
                if phrase in low:
                    snippet = _snippet_around(chunk.text, phrase)
                    signals.append(
                        RiskSignal(
                            signal_type=stype, phrase=phrase,
                            citation=Citation.from_chunk(chunk, score=1.0, quote=snippet),
                            snippet=snippet,
                        )
                    )
                    break  # at most one phrase per signal_type per chunk
    return signals
