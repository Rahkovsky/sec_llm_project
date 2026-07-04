"""Claim-provenance DAG: evidence spans -> atomic claims -> conclusion.

Built entirely from what the pipeline already emits (an ``Answer`` with a
verification report attached) — no re-judging, no fabricated nodes. The graph
makes RAG errors inspectable: which filing passages support which claims, which
claims lack evidence, and why a conclusion was rejected (abstention).

Node types: question | evidence_span | atomic_claim | final_conclusion |
rejected_conclusion.  Edge types: supports | partially_supports | contradicts |
part_of_answer | answers.  A claim with no evidence edge is a missing-evidence
claim (flagged in the mistake table rather than given a fake edge).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Verdict -> edge type (evidence -> claim). UNSUPPORTED yields no edge.
_EDGE_FOR_VERDICT = {"SUPPORTED": "supports", "PARTIAL": "partially_supports",
                     "CONTRADICTED": "contradicts"}
# Default per-claim error weights (Phase 3 replaces these with gold weights):
# a contradiction is worse than a missing-evidence claim is worse than partial.
ERROR_WEIGHTS = {"SUPPORTED": 0.0, "PARTIAL": 1.0, "UNSUPPORTED": 2.0, "CONTRADICTED": 3.0}

_COLORS = {"question": "white", "evidence_span": "lightsteelblue",
           "final_conclusion": "palegreen", "rejected_conclusion": "lightcoral",
           "SUPPORTED": "palegreen", "PARTIAL": "moccasin",
           "UNSUPPORTED": "lightcoral", "CONTRADICTED": "indianred"}


@dataclass
class Node:
    id: str
    type: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    type: str


@dataclass
class ProvenanceDAG:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def node(self, node_id: str) -> Node | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    # ------------------------------------------------------------- exports
    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [asdict(n) for n in self.nodes],
                "edges": [asdict(e) for e in self.edges]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_networkx(self):  # optional dependency, only imported here
        import networkx as nx

        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.id, type=n.type, text=n.text, **n.meta)
        for e in self.edges:
            g.add_edge(e.source, e.target, type=e.type)
        return g

    def to_dot(self, wrap: int = 46) -> str:
        """Graphviz DOT (pure string; render with `dot -Tpng`)."""
        def label(n: Node) -> str:
            words, lines, cur = n.text.split(), [], ""
            for w in words:
                cur = f"{cur} {w}".strip()
                if len(cur) >= wrap:
                    lines.append(cur)
                    cur = ""
            lines.append(cur)
            head = {"evidence_span": n.meta.get("source", ""),
                    "atomic_claim": n.meta.get("verdict", "")}.get(n.type, n.type)
            body = "\\n".join(line.replace('"', "'") for line in lines if line)[:400]
            return f"{head}\\n{body}" if head else body

        out = ["digraph provenance {", '  rankdir=LR; node [shape=box, style=filled];']
        for n in self.nodes:
            color = _COLORS.get(n.meta.get("verdict", ""), _COLORS.get(n.type, "white"))
            out.append(f'  "{n.id}" [label="{label(n)}", fillcolor="{color}"];')
        for e in self.edges:
            style = ' [color=red]' if e.type == "contradicts" else ""
            out.append(f'  "{e.source}" -> "{e.target}" [label="{e.type}"]{style};')
        out.append("}")
        return "\n".join(out)

    # ------------------------------------------------------------ analysis
    def mistake_table(self) -> list[dict[str, Any]]:
        """One row per claim: verdict, weight, evidence source, quote validity."""
        rows = []
        for n in self.nodes:
            if n.type != "atomic_claim":
                continue
            ev = next((e for e in self.edges if e.target == n.id), None)
            src = self.node(ev.source).meta.get("source", "") if ev else ""
            rows.append({
                "claim": n.text,
                "verdict": n.meta.get("verdict", ""),
                "weight": ERROR_WEIGHTS.get(n.meta.get("verdict", ""), 0.0),
                "evidence": src,
                "edge": ev.type if ev else "missing_evidence",
                "valid_quote": n.meta.get("valid_quote", False),
                "numeric_ok": n.meta.get("numeric_ok", True),
            })
        return rows

    def summary(self) -> dict[str, Any]:
        rows = self.mistake_table()
        n = len(rows)
        supported = sum(1 for r in rows if r["verdict"] == "SUPPORTED")
        rejected = [x for x in self.nodes if x.type == "rejected_conclusion"]
        return {
            "claims": n,
            "claim_support_rate": round(supported / n, 3) if n else None,
            "weighted_claim_error_rate": round(sum(r["weight"] for r in rows) / n, 3) if n else None,
            "missing_evidence_claims": sum(1 for r in rows if r["edge"] == "missing_evidence"),
            "contradicted_claims": sum(1 for r in rows if r["verdict"] == "CONTRADICTED"),
            "rejected_conclusion": rejected[0].meta.get("reason") if rejected else None,
        }


def build_dag(answer: dict[str, Any]) -> ProvenanceDAG:
    """Build the DAG from ``Answer.to_dict()`` (verification must be enabled).

    An answer that abstained *before* generation has no claims to graph: its DAG
    is question + rejected_conclusion + the (insufficient) retrieved evidence.
    """
    ver = answer.get("extra", {}).get("verification") or {}
    if not ver and not answer.get("abstained"):
        raise ValueError("answer has no verification report — run with "
                         "verification.enabled: true (the DAG is not built from guesses)")

    dag = ProvenanceDAG()
    dag.nodes.append(Node("q", "question", answer["question"]))

    concl_type = "rejected_conclusion" if answer.get("abstained") else "final_conclusion"
    concl_meta = {"confidence": answer.get("confidence"), "model": answer.get("model")}
    if answer.get("abstained"):
        concl_meta["reason"] = answer.get("abstain_reason", "")
    dag.nodes.append(Node("a", concl_type, answer.get("text", ""), concl_meta))
    dag.edges.append(Edge("a", "q", "answers"))

    evidence = ver.get("evidence") or answer.get("citations", [])
    for i, ev in enumerate(evidence, 1):
        src = f"{ev['ticker']} {ev['filing_type']} {ev['filing_date']} item {ev['item_number']}"
        dag.nodes.append(Node(f"e{i}", "evidence_span", ev["quote"],
                              {"source": src, **{k: ev[k] for k in
                               ("chunk_id", "ticker", "filing_type", "filing_date",
                                "section_title", "item_number", "source_url", "score")}}))

    for j, cv in enumerate(ver.get("claims", []), 1):
        cid = f"c{j}"
        dag.nodes.append(Node(cid, "atomic_claim", cv["claim"],
                              {"verdict": cv["verdict"], "quote": cv.get("quote", ""),
                               "valid_quote": cv.get("valid_quote", False),
                               "numeric_ok": cv.get("numeric_ok", True)}))
        dag.edges.append(Edge(cid, "a", "part_of_answer"))
        edge_type = _EDGE_FOR_VERDICT.get(cv["verdict"])
        idx = cv.get("citation")
        if edge_type and idx and dag.node(f"e{idx}"):
            dag.edges.append(Edge(f"e{idx}", cid, edge_type))
    return dag
