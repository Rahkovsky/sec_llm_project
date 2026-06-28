"""JSON schemas for structured extraction targets (plan item 5).

Each schema describes a list of findings with a short label, a verbatim/near-
verbatim ``evidence`` quote, and the source ``item`` it came from. Keeping the
schemas as plain dicts means no pydantic dependency is required and the same
schema can be handed to any backend that supports JSON-schema-constrained
decoding (OpenAI) or used as an instruction (Anthropic/Ollama).
"""

from __future__ import annotations

from typing import Any


def _finding_array(item_props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": item_props,
            "required": required,
            "additionalProperties": False,
        },
    }


_EVIDENCE = {"type": "string", "description": "Verbatim supporting quote from the filing."}
_SECTION = {"type": "string", "description": "Source item/section, e.g. 'Item 1A'."}


EXTRACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "risk_factors": {
        "type": "object",
        "properties": {
            "risk_factors": _finding_array(
                {
                    "risk": {"type": "string", "description": "Concise risk name."},
                    "category": {"type": "string"},
                    "evidence": _EVIDENCE,
                    "section": _SECTION,
                },
                ["risk", "evidence"],
            )
        },
        "required": ["risk_factors"],
        "additionalProperties": False,
    },
    "litigation": {
        "type": "object",
        "properties": {
            "litigation": _finding_array(
                {
                    "matter": {"type": "string"},
                    "parties": {"type": "string"},
                    "status": {"type": "string"},
                    "evidence": _EVIDENCE,
                    "section": _SECTION,
                },
                ["matter", "evidence"],
            )
        },
        "required": ["litigation"],
        "additionalProperties": False,
    },
    "mdna": {
        "type": "object",
        "properties": {
            "highlights": _finding_array(
                {
                    "topic": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence": _EVIDENCE,
                    "section": _SECTION,
                },
                ["topic", "evidence"],
            )
        },
        "required": ["highlights"],
        "additionalProperties": False,
    },
    "internal_controls": {
        "type": "object",
        "properties": {
            "controls": _finding_array(
                {
                    "assessment": {"type": "string"},
                    "material_weakness": {"type": "boolean"},
                    "evidence": _EVIDENCE,
                    "section": _SECTION,
                },
                ["assessment", "evidence"],
            )
        },
        "required": ["controls"],
        "additionalProperties": False,
    },
    "related_party": {
        "type": "object",
        "properties": {
            "transactions": _finding_array(
                {
                    "counterparty": {"type": "string"},
                    "description": {"type": "string"},
                    "evidence": _EVIDENCE,
                    "section": _SECTION,
                },
                ["description", "evidence"],
            )
        },
        "required": ["transactions"],
        "additionalProperties": False,
    },
    "liquidity": {
        "type": "object",
        "properties": {
            "concerns": _finding_array(
                {
                    "concern": {"type": "string"},
                    "severity": {"type": "string"},
                    "evidence": _EVIDENCE,
                    "section": _SECTION,
                },
                ["concern", "evidence"],
            )
        },
        "required": ["concerns"],
        "additionalProperties": False,
    },
}

# Default section to retrieve from for each extraction target (item number).
PREFERRED_ITEMS: dict[str, str] = {
    "risk_factors": "1A",
    "litigation": "3",
    "mdna": "7",
    "internal_controls": "9A",
    "related_party": "13",
    "liquidity": "7",
}


def get_schema(target: str) -> dict[str, Any]:
    try:
        return EXTRACTION_SCHEMAS[target]
    except KeyError as exc:
        raise KeyError(
            f"Unknown extraction target '{target}'. "
            f"Available: {', '.join(sorted(EXTRACTION_SCHEMAS))}"
        ) from exc
