"""Dropped qualifiers: a claim that states as fact what the filing hedged.

10-K language is hedged on purpose ("we believe", "may", "could", "is expected
to"). An answer that turns "competition may further intensify" into "competition
is intensified", or "the impact is not expected to be material" into "there was no
material impact", is supported word for word but says something the filing does
not.

Detection has two stages:

1. ``is_candidate`` — the lexical proxy used by tools/provenance_report.py: the
   supporting quote contains a hedge and the claim contains none. Cheap, high
   recall, low precision: of the 73 claims it flagged in the v3 run, 14 were real
   when hand-labelled (eval/qualifier_labels.json). Most hits are hedges in another
   clause ("claims that *may* arise"), adjectives ("*potential* vulnerabilities")
   or "*subject to* competition". A clause-anchored lexical rule reached only
   precision 0.50 / recall 0.50: real cases are paraphrased ("We believe that
   competitive performance ... depends upon" -> "Success depends on").
2. ``QualifierChecker`` — the independent judge model decides, for the
   candidates only, whether the claim drops a qualifier that applies to what it
   states. One short call per answer with candidates.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..llm.base import LLMProvider, Message

log = logging.getLogger(__name__)

# Modal and forward-looking/opinion hedges (same list as the report's proxy).
HEDGES = ("may", "could", "might", "would", "believe", "believes", "believed",
          "expect", "expects", "expected", "anticipate", "anticipates", "anticipated",
          "intend", "intends", "intended", "potential", "potentially", "likely",
          "subject to")
_HEDGE_RE = re.compile(r"\b(" + "|".join(HEDGES) + r")\b", re.IGNORECASE)


def hedges_in(text: str) -> set[str]:
    return {h.lower() for h in _HEDGE_RE.findall(text)}


def is_candidate(claim: str, quote: str) -> list[str]:
    """Hedges in ``quote`` when ``claim`` keeps none (the lexical proxy)."""
    ev = hedges_in(quote)
    return sorted(ev) if ev and not hedges_in(claim) else []


QUALIFIER_SYSTEM = (
    "You check whether a summary sentence keeps the qualifiers of the SEC filing "
    "text it is based on. You answer with one line per item and nothing else.")
QUALIFIER_PROMPT = (
    "For each numbered item, compare the CLAIM with the filing QUOTE.\n"
    "Answer DROPPED if the quote qualifies the very thing the claim states (with "
    "words like may, could, might, believe, expect, anticipate, intend, potential, "
    "likely) and the claim states it as a plain fact or a certainty. Examples: "
    "\"competition may further intensify\" -> \"competition is intensified\"; "
    "\"we believe our products lead the market\" -> \"its products lead the market\"; "
    "\"could harm our business\" -> \"can harm the business\"; \"is not expected to be "
    "material\" -> \"was not material\".\n"
    "Answer OK if the claim keeps the qualifier or attributes the statement to the "
    "company (\"the company believes/states/expects ...\"), if the qualifier in the "
    "quote applies to a different part of the sentence than what the claim states "
    "(\"liabilities that may arise\" when the claim is about the indemnity itself), "
    "or if the word is not a qualifier there (\"potential vulnerabilities\", "
    "\"subject to competition\", \"expected cost\").\n"
    "Reply with lines of the form \"<number>: DROPPED\" or \"<number>: OK\".\n\n"
    "{items}")
# v2: make the judge locate the hedge that governs the claimed fact before deciding.
# v1 (above) was precise but rarely said DROPPED (recall 2/14 on the hand labels).
QUALIFIER_PROMPT_V2 = (
    "For each numbered item, do three steps.\n"
    "(a) Find the words in the QUOTE that say what the CLAIM states.\n"
    "(b) Are those words qualified by a hedge such as may, could, might, would, "
    "believe, expect, anticipate, intend, potential, likely, or \"not expected to\"? "
    "Only count a hedge that governs the stated fact itself, not one in a different "
    "clause (\"claims that may arise\" does not hedge the indemnity) and not an "
    "adjective (\"potential vulnerabilities\", \"subject to competition\").\n"
    "(c) Does the CLAIM keep that hedge (same or equivalent word) or attribute the "
    "statement to the company (\"the company believes/expects/states\")?\n"
    "The item is DROPPED if (b) found a hedge and (c) is no. Examples of DROPPED: "
    "\"could harm our business\" -> \"can harm the business\"; \"We believe that our "
    "performance depends upon innovation\" -> \"Success depends on innovation\"; "
    "\"is not expected to be material\" -> \"did not have a material impact\"; "
    "\"competition may further intensify\" -> \"competition has intensified\".\n"
    "Reply with one line per item and nothing else, in the form\n"
    "<number>: hedge=<the hedge word or none>; kept=<yes or no>; <DROPPED or OK>\n\n"
    "{items}")
_LINE_V2_RE = re.compile(r"^[\s*#-]*\[?(\d+)\]?\**\s*[:.)-].*?\b(DROPPED|OK)\b\**\s*$",
                         re.IGNORECASE | re.M)
_LINE_RE = re.compile(r"^[\s*#-]*\[?(\d+)\]?\**\s*[:.)-]\s*\**\s*(DROPPED|OK)\b",
                      re.IGNORECASE | re.M)


@dataclass(frozen=True)
class QualifierItem:
    claim: str
    quote: str


class QualifierChecker:
    """Judge-model check of lexical candidates (one call for all of an answer's)."""

    def __init__(self, llm: LLMProvider, max_tokens: int = 400, prompt: str = "v2") -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.prompt = prompt

    def check(self, items: list[QualifierItem]) -> list[bool | None]:
        """True = qualifier dropped, False = OK, None = no usable answer."""
        if not items:
            return []
        body = "\n\n".join(f"{i}. CLAIM: {it.claim}\n   QUOTE: {it.quote}"
                           for i, it in enumerate(items, 1))
        try:
            text = self.llm.generate(
                [Message("system", QUALIFIER_SYSTEM),
                 Message("user", (QUALIFIER_PROMPT_V2 if self.prompt == "v2"
                                  else QUALIFIER_PROMPT).format(items=body))],
                temperature=0.0, max_tokens=self.max_tokens).text
        except Exception as exc:  # a failed check must not fail the answer
            log.warning("qualifier check failed: %s", exc)
            return [None] * len(items)
        return parse_check(text, len(items), self.prompt)


def parse_check(text: str, n: int, prompt: str = "v1") -> list[bool | None]:
    out: list[bool | None] = [None] * n
    for m in (_LINE_V2_RE if prompt == "v2" else _LINE_RE).finditer(text):
        k = int(m.group(1))
        if 1 <= k <= n:
            out[k - 1] = m.group(2).upper() == "DROPPED"
    return out
