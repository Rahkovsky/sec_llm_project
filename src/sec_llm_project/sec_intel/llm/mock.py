"""Deterministic, offline mock LLM (default backend).

This backend requires no network or model weights, which keeps the platform
runnable and unit-testable in CI (plan item 11). It is intentionally simple but
"useful enough" to drive the end-to-end pipeline:

* For free-form generation it returns an extractive answer built from the most
  question-relevant sentences found in the prompt's CONTEXT block.
* For JSON-schema-constrained requests it emits a valid instance of the schema,
  filling string fields with relevant snippets from the context.

Being deterministic (no randomness) it satisfies the "deterministic prompts"
requirement (plan item 10) and yields stable evaluation numbers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import GenerationResult, Message

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "are", "was", "were", "be", "by", "as", "at", "that", "this", "from", "it",
    "its", "we", "our", "their", "have", "has", "had", "which", "these", "those",
    "what", "how", "did", "does", "do", "than", "into", "about",
}


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


class MockLLM:
    """Deterministic extractive responder implementing ``LLMProvider``."""

    def __init__(self, model: str = "mock-1") -> None:
        self.name = model
        self.model = model

    # ------------------------------------------------------------------ API
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        user = "\n\n".join(m.content for m in messages if m.role == "user")
        question = self._extract_marker(user, "QUESTION") or self._first_question(user)
        context = self._extract_marker(user, "CONTEXT") or user

        if json_schema is not None:
            obj = _instance_from_schema(json_schema, context)
            text = json.dumps(obj, ensure_ascii=False, indent=2)
            return GenerationResult(text=text, model=self.model, finish_reason="stop")

        text = self._grounded_answer(question, context, max_sentences=4)
        return GenerationResult(text=text, model=self.model, finish_reason="stop")

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _extract_marker(text: str, marker: str) -> str:
        m = re.search(rf"{marker}:\s*(.*?)(?:\n[A-Z]{{3,}}:|\Z)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _first_question(text: str) -> str:
        for line in text.splitlines():
            if line.strip().endswith("?"):
                return line.strip()
        return text[:200]

    def _grounded_answer(self, question: str, context: str, *, max_sentences: int) -> str:
        """Pick the most relevant sentences and tag them with their excerpt number.

        Returns ``INSUFFICIENT_EVIDENCE`` when the question shares no terms with
        any excerpt, so abstention works without a real model.
        """
        qwords = _keywords(question)
        blocks = self._numbered_blocks(context)
        scored: list[tuple[float, int, str]] = []  # (score, block_no, sentence)
        for block_no, body in blocks:
            for sent in _split_sentences(body):
                swords = _keywords(sent)
                if not swords:
                    continue
                overlap = len(qwords & swords)
                if overlap <= 0:
                    continue
                length_penalty = 1.0 + abs(len(swords) - 18) / 40.0
                scored.append((overlap / length_penalty, block_no, sent))
        if not scored:
            return "INSUFFICIENT_EVIDENCE"
        scored.sort(key=lambda t: -t[0])
        chosen: list[tuple[float, int, str]] = []
        seen_sents: set[str] = set()
        for item in scored:
            key = item[2].strip().lower()
            if key in seen_sents:
                continue
            seen_sents.add(key)
            chosen.append(item)
            if len(chosen) >= max_sentences:
                break
        cited = sorted({bn for _, bn, _ in chosen})
        body = " ".join(sent for _, _, sent in chosen)
        cites = " ".join(f"[{n}]" for n in cited)
        return f"{body} {cites}".strip()

    @staticmethod
    def _numbered_blocks(context: str) -> list[tuple[int, str]]:
        """Parse a ``[1] ... [2] ...`` context into (number, body) pairs."""
        matches = list(re.finditer(r"\[(\d+)\]", context))
        if not matches:
            return [(1, context)]
        blocks: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
            blocks.append((int(m.group(1)), context[start:end].strip()))
        return blocks


# --------------------------------------------------------------------- schema
def _instance_from_schema(schema: dict[str, Any], context: str) -> Any:
    """Build a minimal valid instance of a (subset of) JSON Schema.

    Supports object/array/string/number/boolean which covers the extraction
    schemas defined in :mod:`sec_intel.extraction.schemas`.
    """
    stype = schema.get("type")
    if stype == "object":
        props: dict[str, Any] = schema.get("properties", {})
        out: dict[str, Any] = {}
        for key, subschema in props.items():
            out[key] = _instance_from_schema(subschema, context)
        return out
    if stype == "array":
        items = schema.get("items", {"type": "string"})
        # Heuristic: pull up to 3 candidate sentences relevant to the field.
        candidates = _split_sentences(context)[:3]
        if items.get("type") == "object":
            return [_instance_from_schema(items, c) for c in candidates] if candidates else []
        return list(candidates[:3])
    if stype == "string":
        sentences = _split_sentences(context)
        return sentences[0] if sentences else ""
    if stype in {"number", "integer"}:
        return 0
    if stype == "boolean":
        return False
    return None
