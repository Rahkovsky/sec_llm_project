"""Drop the answer sentences that carry claims the judge could not support.

The verifier labels every atomic claim of an answer. When a claim is UNSUPPORTED or
CONTRADICTED, the sentence it came from is removed from the answer shown to the user;
a supported claim in the same sentence goes with it. Meta-claims ("the excerpts do
not say X") are never removed: they tell the reader what the filings leave open.

Claims are mapped to sentences by word overlap, because the judge rewrites claims
rather than quoting the answer. See docs/EXPERIMENTS.md §20.1 (Q2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.types import CONTRADICTED, UNSUPPORTED, ClaimVerdict

FAIL = {UNSUPPORTED, CONTRADICTED}
_WORD = re.compile(r"[a-z0-9]+")
# a sentence ends at . ! ? followed by space and a capital, bracket or quote; or at a newline
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'“])|\n+")


def _words(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


# a piece ending like this is not a sentence end: an initial ("Jonathan P."), a title or
# a company suffix
# (also short capitalized abbreviations in case names: "Waterford Twp. Gen. Emps. Ret. Sys. v.")
_ABBREV = re.compile(r"(?:\b[A-Z][a-z]{0,3}|\b(?:Corp|Inc|Ltd|Jr|Sr|vs?|U\.S))\.$")


def split_sentences(text: str) -> list[str]:
    """Sentences, each keeping its trailing citation such as ' [2]'."""
    out: list[str] = []
    for piece in (s.strip() for s in _SENT.split(text)):
        if not piece:
            continue
        if out and _ABBREV.search(out[-1]) and not out[-1].endswith("\n"):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def is_lead_in(sentence: str) -> bool:
    """A line that asserts nothing itself: 'Regarding storage:', 'The framework includes
    the following:', a markdown heading ('**Risk Profile**', '*   **Technology:**') or a
    bare label ('AWK'), with no citation."""
    if "[" in sentence:
        return False
    core = sentence.strip().strip("*#-• ").strip()
    return core.endswith(":") or (len(core.split()) <= 6 and not core.endswith((".", "!", "?")))


def owner_sentence(claim: str, sentences: list[str]) -> int:
    """Index of the sentence that shares the largest share of the claim's words."""
    cw = _words(claim)
    scores = [len(cw & _words(s)) / (len(cw) or 1) for s in sentences]
    return max(range(len(sentences)), key=scores.__getitem__)


@dataclass
class PruneResult:
    text: str
    kept: list[ClaimVerdict]
    removed_sentences: list[str] = field(default_factory=list)
    removed_claims: list[ClaimVerdict] = field(default_factory=list)
    collateral: int = 0  # non-failing claims removed because they shared a sentence

    @property
    def changed(self) -> bool:
        return bool(self.removed_sentences)


def prune_unsupported(text: str, claims: list[ClaimVerdict]) -> PruneResult:
    sentences = split_sentences(text)
    if not sentences or not claims:
        return PruneResult(text=text, kept=list(claims))
    owner = [owner_sentence(c.claim, sentences) for c in claims]
    fails = [c.verdict in FAIL and not c.meta for c in claims]
    drop = {owner[i] for i, f in enumerate(fails) if f}
    if not drop:
        return PruneResult(text=text, kept=list(claims))
    return PruneResult(
        text=" ".join(s for i, s in enumerate(sentences) if i not in drop),
        kept=[c for i, c in enumerate(claims) if owner[i] not in drop],
        removed_sentences=[sentences[i] for i in sorted(drop)],
        removed_claims=[c for i, c in enumerate(claims) if owner[i] in drop],
        collateral=sum(not fails[i] for i in range(len(claims)) if owner[i] in drop),
    )
