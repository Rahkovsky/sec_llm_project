"""Structured extraction over retrieved evidence (plan item 5).

Retrieves the most relevant passages for a target (optionally constrained to the
canonical SEC item), asks the LLM to emit JSON conforming to the target schema,
and parses/validates the result. Each finding stays attributable because the
schema requires a verbatim ``evidence`` quote and the source passages are
recorded alongside the output (plan items 8, 10).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.audit import AuditLogger, null_logger
from ..core.types import Citation, RetrievalResult
from ..llm.base import LLMProvider, Message
from ..retrieval.hybrid import HybridRetriever
from .schemas import PREFERRED_ITEMS, get_schema


@dataclass
class ExtractionResult:
    target: str
    data: dict[str, Any]
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    valid: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "data": self.data,
            "citations": [c.to_dict() for c in self.citations],
            "model": self.model,
            "valid": self.valid,
            "error": self.error,
        }


def _extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating code fences / preamble."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _validate(data: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    """Lightweight structural validation against the schema's required keys."""
    if not isinstance(data, dict):
        return False, "expected a JSON object at the top level"
    for key in schema.get("required", []):
        if key not in data:
            return False, f"missing required key '{key}'"
        if schema["properties"][key].get("type") == "array" and not isinstance(data[key], list):
            return False, f"key '{key}' must be an array"
    return True, ""


class StructuredExtractor:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider,
                 audit: AuditLogger | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.audit = audit or null_logger()

    def extract(self, target: str, *, ticker: str | None = None,
                filing_type: str | None = None, top_k: int = 8,
                constrain_to_item: bool = True) -> ExtractionResult:
        schema = get_schema(target)
        filters: dict[str, Any] = {}
        if ticker:
            filters["ticker"] = ticker.upper()
        if filing_type:
            filters["filing_type"] = filing_type
        if constrain_to_item and target in PREFERRED_ITEMS:
            filters["item_number"] = PREFERRED_ITEMS[target]

        query = target.replace("_", " ")
        results = self.retriever.retrieve(query, top_k=top_k, filters=filters or None)
        if not results and "item_number" in filters:
            # Relax the item constraint if the section wasn't detected.
            filters.pop("item_number")
            results = self.retriever.retrieve(query, top_k=top_k, filters=filters or None)

        context = self._format_context(results)
        messages = self._build_messages(target, schema, context)
        gen = self.llm.generate(messages, json_schema=schema, temperature=0.0)

        citations = [Citation.from_result(r) for r in results]
        try:
            data = _extract_json(gen.text)
            valid, err = _validate(data, schema)
        except (json.JSONDecodeError, ValueError) as exc:
            data, valid, err = {}, False, f"JSON parse error: {exc}"

        self.audit.log(
            "extraction",
            {"target": target, "filters": filters, "valid": valid,
             "model": gen.model, "n_evidence": len(results)},
        )
        return ExtractionResult(
            target=target, data=data, citations=citations,
            model=gen.model, valid=valid, error=err,
        )

    @staticmethod
    def _format_context(results: list[RetrievalResult]) -> str:
        blocks = []
        for i, r in enumerate(results, 1):
            m = r.chunk.metadata
            blocks.append(
                f"[{i}] ({m.ticker} {m.filing_type} {m.filing_date}, "
                f"{r.chunk.section_title}) {r.chunk.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_messages(target: str, schema: dict[str, Any], context: str) -> list[Message]:
        system = (
            "You are a meticulous SEC filing analyst. Extract structured facts "
            "ONLY from the provided excerpts. Every finding must include a "
            "verbatim supporting quote in its 'evidence' field. Do not invent "
            "facts. Respond with JSON conforming exactly to the provided schema."
        )
        user = (
            f"TARGET: {target}\n"
            f"SCHEMA: {json.dumps(schema)}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Return only the JSON object."
        )
        return [Message("system", system), Message("user", user)]
