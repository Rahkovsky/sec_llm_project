"""Locks the qualifier-preservation heuristic against the bugs found while
validating it on real filings: substring matches, cross-hedges, topic words."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from provenance_report import hardened_hedges, hedges_in


def test_word_boundary_no_substring_false_positives():
    # "difficult" must not match "if"; "uncertain" must not match "certain".
    assert hedges_in("this is difficult and uncertain") == set()
    assert hedges_in("results could adversely affect us") == {"could"}
    assert hedges_in("costs are subject to change") == {"subject to"}


def test_hardening_flagged_only_when_claim_keeps_no_hedge():
    # Bare fact from hedged evidence -> flagged.
    assert hardened_hedges("Competition exists.",
                           "competitive pressure could adversely affect us") == ["could"]
    # Claim keeps its own hedge (may) -> not a drop even if evidence used could.
    assert hardened_hedges("Competitors may offer more.",
                           "any competitor could deliver more") == []
    # Evidence had no hedge -> nothing to drop.
    assert hardened_hedges("Revenue rose 5%.", "net sales increased 5%") == []


def test_forward_looking_variants_recognized_both_sides():
    # -ed variant in the claim counts as keeping the hedge.
    assert hardened_hedges("Cash flows are expected to grow.",
                           "the company expects cash flows to grow") == []
