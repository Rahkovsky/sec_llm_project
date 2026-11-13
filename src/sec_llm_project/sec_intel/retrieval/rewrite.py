"""Query expansion with a hypothetical filing passage (HyDE).

Analyst questions are terse ("Cap allocation impact on investor confidence,
MCD") and rarely use the vocabulary of the 10-K section that answers them.
The generator writes a short passage in 10-K style that *would* answer the
question; retrieval then scores each chunk by the mean of its cosine to the
original question and to that passage (both embedded as queries).

The generator sees the question **company-blind**: every mention of the company
(name, ticker, short alias) is replaced by "the company", and the prompt names
no company. The retrieval filter already scopes the search to the company, so
the passage only has to supply the vocabulary. Naming the company made
gemma4:12b refuse ("Please provide the text of the 10-K...") on 168 of 361
natural, specific questions, and averaging a refusal into the query vector
cost 7 points of Hit@8; company-blind passages refused on 1 of 361. Without a
name or a ticker, the model also cannot expand a ticker as an abbreviation
("ES" -> "Environmental Settlements") or recall facts about the company.

Hit@8 with MMR (tools/rewrite_lab.py, Nov 2026), question -> question + passage:
FinDER (248) 0.593 -> 0.758 company-blind (0.774 with the company named);
Claude-written natural questions (361) 0.825 -> 0.972 company-blind
(0.753 with the company named). One LLM call per question (~5-8 s with
gemma4:12b on an M1 Pro).

The passage is a retrieval key only: it is never shown to the generator or
used as evidence, so invented content in it cannot reach an answer. Output
that looks like a refusal is discarded (retrieval falls back to the question).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from ..llm.base import LLMProvider, Message

log = logging.getLogger(__name__)

SYSTEM = ("You help search a company's SEC 10-K annual report. You never answer from "
          "memory and never add facts that are not in the user's text.")
HYDE_PROMPT = (
    "Write a short passage (3-4 sentences) in the formal style of a 10-K annual report "
    "that would answer the question below. Use the wording companies typically use in "
    "the relevant 10-K section (for example Risk Factors, Legal Proceedings, "
    "Cybersecurity, MD&A, or the notes to the financial statements). Do not invent "
    "specific numbers or names; keep it generic. Output only the passage.\n\n"
    "Question: {q}")
_REFUSAL = re.compile(r"(?i)\b(please provide|i cannot|i can't|i am unable|i'm unable|"
                      r"i do not have|i don't have|no text (?:was|is) provided|not provided)")


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text[:200]))


class HydeExpander:
    """question (+ ticker filter) -> [hypothetical 10-K passage], company-blind."""

    def __init__(self, llm: LLMProvider, mask: Callable[[str, list[str]], str] | None = None,
                 max_tokens: int = 300) -> None:
        self.llm = llm
        self.mask = mask or (lambda q, tickers: q)
        self.max_tokens = max_tokens
        self._cache: dict[str, list[str]] = {}

    def expand(self, query: str, filters: dict[str, Any] | None = None) -> list[str]:
        t = (filters or {}).get("ticker")
        tickers = [t] if isinstance(t, str) else list(t or []) if not isinstance(t, dict) \
            else list(t.get("$in", []))
        blind = self.mask(query, tickers)
        if blind in self._cache:
            return self._cache[blind]
        try:
            text = self.llm.generate(
                [Message("system", SYSTEM), Message("user", HYDE_PROMPT.format(q=blind))],
                temperature=0.0, max_tokens=self.max_tokens).text.strip()
        except Exception as exc:  # retrieval must not fail because expansion did
            log.warning("HyDE expansion failed, using the question alone: %s", exc)
            return []
        if is_refusal(text):
            log.info("HyDE passage looks like a refusal, using the question alone: %.80s", text)
            text = ""
        out = [text] if text else []
        self._cache[blind] = out
        return out
