"""Deterministic prompt templates (plan item 10).

Prompts are fixed strings with explicit placeholders so that, combined with a
zero-temperature provider, generation is reproducible and auditable.
"""

from __future__ import annotations

from ..core.types import RetrievalResult

SYSTEM_PROMPT = (
    "You are a precise SEC filing analyst. Answer ONLY using the numbered "
    "excerpts provided. Cite the excerpts you rely on using bracketed numbers "
    "like [1] or [2]. If the excerpts do not contain enough information to "
    "answer, reply exactly: INSUFFICIENT_EVIDENCE. Be concise, factual, and "
    "never speculate beyond the excerpts."
)


def format_context(results: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        m = r.chunk.metadata
        header = (
            f"[{i}] {m.company if m.company != 'UNKNOWN' else m.ticker} "
            f"{m.filing_type} {m.filing_date} — {r.chunk.section_title} "
            f"(Item {r.chunk.item_number})"
        )
        blocks.append(f"{header}\n{r.chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{format_context(results)}\n\n"
        "Provide a grounded answer with bracketed citations."
    )


# --------------------------------------------------------------- verification
JUDGE_SYSTEM_PROMPT = (
    "You are an independent verifier of answers about SEC filings. You did NOT "
    "write the answer. Break the ANSWER into atomic factual claims. For EACH claim, "
    "judge it ONLY against the numbered EVIDENCE. Respond with JSON: an object with "
    "key 'claims' whose value is a list of objects {claim, verdict, citation, quote}. "
    "'verdict' is exactly one of SUPPORTED, PARTIAL, UNSUPPORTED, CONTRADICTED. "
    "For a SUPPORTED or PARTIAL claim, 'citation' MUST be the number of the "
    "evidence passage that supports it and 'quote' a span copied VERBATIM from "
    "that passage — a claim you cannot tie to a specific numbered passage is "
    "UNSUPPORTED (citation null, quote empty). "
    "Never infer beyond the evidence: mark a claim absent from the evidence as "
    "UNSUPPORTED. Use CONTRADICTED only when a specific evidence passage directly "
    "refutes the claim — then 'citation' MUST be that evidence number and 'quote' "
    "the refuting span. Absence of evidence is UNSUPPORTED, never CONTRADICTED."
)

# Minimal JSON Schema for schema-constrained providers (and the offline mock).
JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": [
                        "SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]},
                    "citation": {"type": ["integer", "null"]},
                    "quote": {"type": "string"},
                },
                "required": ["claim", "verdict", "citation"],
            },
        }
    },
}


def build_judge_prompt(answer_text: str, results: list[RetrievalResult]) -> str:
    return (
        f"ANSWER:\n{answer_text}\n\n"
        f"EVIDENCE:\n{format_context(results)}\n\n"
        "Return ONLY the JSON verdict object."
    )


def build_repair_prompt(answer_text: str, flagged_claims: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in flagged_claims)
    return (
        "An independent verifier found these claims are NOT supported by the "
        f"numbered excerpts:\n{bullets}\n\n"
        "Revise your previous answer to remove or correct every unsupported claim, "
        "keeping only statements directly supported by the excerpts and retaining "
        "their [n] citations. If nothing supportable remains, reply exactly: "
        "INSUFFICIENT_EVIDENCE."
    )
