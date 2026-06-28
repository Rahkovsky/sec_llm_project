"""Evaluation suite: retrieval, citation, hallucination, latency (plan item 9)."""

from __future__ import annotations

from .enforcement import (
    ENFORCEMENT_CASES,
    EnforcementCase,
    load_enforcement_cases,
    run_enforcement_benchmark,
)
from .metrics import average_precision_at_k, mrr, recall_at_k
from .runner import EvalCase, EvalReport, EvaluationRunner

__all__ = [
    "ENFORCEMENT_CASES",
    "EnforcementCase",
    "EvalCase",
    "EvalReport",
    "EvaluationRunner",
    "average_precision_at_k",
    "load_enforcement_cases",
    "mrr",
    "recall_at_k",
    "run_enforcement_benchmark",
]
