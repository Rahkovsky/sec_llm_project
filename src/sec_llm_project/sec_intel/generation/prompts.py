"""Deterministic prompt templates (plan item 10).

Prompts are fixed strings with explicit placeholders so that, combined with a
zero-temperature provider, generation is reproducible and auditable.
"""

from __future__ import annotations

from ..core.types import RetrievalResult

SYSTEM_PROMPT = (
    "You are a precise SEC filing analyst. Answer ONLY using the numbered "
    "excerpts provided. Cite the excerpts you rely on using bracketed numbers "
    "like [1] or [2]. Be concise, factual, and never speculate beyond the excerpts.\n"
    "- If the excerpts answer only part of the question (for example, they describe "
    "a policy or a figure but not its effect on investors, valuation or the stock), "
    "answer that part with citations and say in one sentence which part the excerpts "
    "do not cover. Do not fill the gap with your own inference.\n"
    "- Reply exactly INSUFFICIENT_EVIDENCE, and nothing else, if the excerpts contain "
    "nothing that answers any part of the question, or if the question rests on a "
    "premise the excerpts do not support: a period, document, event or company that "
    "does not appear in them."
)

QUALIFIER_RULE = (
    "\n- Keep the filing's qualifiers. If an excerpt says the company believes, "
    "expects, anticipates or intends something, or that something may, could or is "
    "likely to happen, say so in your answer (\"the company believes ...\", \"... may "
    "...\"). Do not state it as a fact or a certainty."
)


# v5 (Nov 13 2026): with the qualifier rule, v4 answers grew 17% (4.88 -> 5.73
# claims) and unsupported claims grew with them (docs/EXPERIMENTS.md §14).
CONCISE_RULE = (
    "\n- Keep the answer short: one sentence per point, each point once (do not "
    "restate what several excerpts say alike), and leave out background the question "
    "does not ask for."
)


def system_prompt(qualifier_rule: bool = False, concise_rule: bool = False) -> str:
    return (SYSTEM_PROMPT + (QUALIFIER_RULE if qualifier_rule else "")
            + (CONCISE_RULE if concise_rule else ""))


# v1/v2 prompt (abstained on any gap; kept for comparison runs):
# "...If the excerpts do not contain enough information to answer, reply exactly:
#  INSUFFICIENT_EVIDENCE. Be concise, factual, and never speculate beyond the excerpts."


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


def build_qualifier_repair_prompt(dropped: list[tuple[str, str]]) -> str:
    items = "\n".join(f"- CLAIM: {c}\n  FILING TEXT: \"{q}\"" for c, q in dropped)
    return (
        "An independent verifier found that these statements in your answer drop a "
        "qualifier the filing uses (for example \"may\", \"could\", \"believes\", "
        f"\"expects\"), so they state as fact what the filing qualified:\n{items}\n\n"
        "Rewrite your previous answer so that each of these statements keeps the "
        "filing's qualifier. Change nothing else: keep every other sentence and all "
        "[n] citations as they are."
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
