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

# fill / border pairs, shared by the DOT and HTML renderers.
_PALETTE = {"question": ("#f3f4f6", "#6b7280"), "evidence_span": ("#dbeafe", "#3b82f6"),
            "final_conclusion": ("#e0e7ff", "#4f46e5"),
            "rejected_conclusion": ("#fee2e2", "#b91c1c"),
            "SUPPORTED": ("#dcfce7", "#16a34a"), "PARTIAL": ("#fef3c7", "#d97706"),
            "UNSUPPORTED": ("#fee2e2", "#dc2626"), "CONTRADICTED": ("#fecaca", "#991b1b")}
_COLORS = {k: v[0] for k, v in _PALETTE.items()}
_BADGE = {"SUPPORTED": "✓ supported", "PARTIAL": "◐ partial",
          "UNSUPPORTED": "✗ unsupported", "CONTRADICTED": "‼ contradicted"}


def _wrap(text: str, width: int = 34, max_lines: int = 3) -> str:
    lines, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width and cur:
            lines.append(cur)
            cur = word
            if len(lines) == max_lines:
                return "\n".join(lines)[:-1] + "…"
        else:
            cur = f"{cur} {word}".strip()
    lines.append(cur)
    return "\n".join(lines)


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceDAG:
        return cls(nodes=[Node(**n) for n in data["nodes"]],
                   edges=[Edge(**e) for e in data["edges"]])

    def to_networkx(self):  # optional dependency, only imported here
        import networkx as nx

        g = nx.DiGraph()
        for n in self.nodes:
            g.add_node(n.id, type=n.type, text=n.text, **n.meta)
        for e in self.edges:
            g.add_edge(e.source, e.target, type=e.type)
        return g

    def to_dot(self) -> str:
        """Slide-quality Graphviz DOT (render: `dot -Tpdf` / `-Tpng -Gdpi=170`).

        Same visual language as ``to_html``: evidence → claims → conclusion,
        verdict badges, dashed muted style for retrieved-but-uncited evidence.
        The question goes in the figure caption, not the graph.
        """
        cited = {e.source for e in self.edges if e.type in _EDGE_FOR_VERDICT.values()}
        q = self.node("q")
        cap = (q.text.replace('"', "'") if q else "")
        out = ["digraph provenance {",
               '  rankdir=LR; splines=true; nodesep=0.35; ranksep=1.1; bgcolor=white;',
               f'  label="{cap}"; labelloc=t; fontname="Helvetica"; fontsize=15;',
               '  node [shape=box, style="rounded,filled", fontname="Helvetica",'
               ' fontsize=11, penwidth=1.4, margin="0.16,0.1"];',
               '  edge [fontname="Helvetica", fontsize=9, arrowsize=0.7, color="#94a3b8"];']
        for n in self.nodes:
            if n.type == "question":
                continue
            fill, border = _PALETTE.get(n.meta.get("verdict", ""),
                                        _PALETTE.get(n.type, ("#ffffff", "#666666")))
            head = {"evidence_span": n.meta.get("source", "evidence"),
                    "atomic_claim": _BADGE.get(n.meta.get("verdict", ""), "claim"),
                    "final_conclusion": "Conclusion",
                    "rejected_conclusion":
                        f"Rejected — {n.meta.get('reason', '')}"}[n.type]
            style = ', style="rounded,filled,dashed"' if (
                n.type == "evidence_span" and n.id not in cited) else ""
            if style:  # retrieved but uncited: muted
                fill, border = "#f8fafc", "#94a3b8"
            label = (head.replace('"', "'") + r"\n"
                     + _wrap(n.text, 38, 3).replace('"', "'").replace("\n", r"\n"))
            out.append(f'  "{n.id}" [label="{label}", fillcolor="{fill}", '
                       f'color="{border}"{style}];')
        for e in self.edges:
            if e.type == "answers":
                continue
            if e.type in _EDGE_FOR_VERDICT.values():
                border = _PALETTE[
                    {v: k for k, v in _EDGE_FOR_VERDICT.items()}[e.type]][1]
                lbl = e.type.replace("partially_supports", "partial").replace("_", " ")
                out.append(f'  "{e.source}" -> "{e.target}" '
                           f'[label="{lbl}", color="{border}", fontcolor="{border}", penwidth=1.6];')
            else:
                out.append(f'  "{e.source}" -> "{e.target}";')
        out.append("}")
        return "\n".join(out)

    def to_html(self, title: str = "Claim provenance") -> str:
        """Presentation-quality interactive HTML (vis-network via CDN, no py deps).

        Reading order left→right: evidence → claims → conclusion. The question is
        shown in the header (not as a node); uncited evidence renders muted with a
        dashed border; hover any node for the full span + provenance.
        """
        level = {"evidence_span": 0, "atomic_claim": 1,
                 "final_conclusion": 2, "rejected_conclusion": 2}
        cited = {e.source for e in self.edges if e.type in _EDGE_FOR_VERDICT.values()}

        vnodes = []
        for n in self.nodes:
            if n.type == "question":
                continue  # lives in the header
            fill, border = _PALETTE.get(n.meta.get("verdict", ""),
                                        _PALETTE.get(n.type, ("#fff", "#666")))
            head = {"evidence_span": n.meta.get("source", "evidence"),
                    "atomic_claim": _BADGE.get(n.meta.get("verdict", ""), "claim"),
                    "final_conclusion": "Conclusion",
                    "rejected_conclusion": f"Rejected — {n.meta.get('reason', '')}"}[n.type]
            node = {"id": n.id, "level": level[n.type],
                    "label": f"*{head}*\n{_wrap(n.text)}",
                    "title": n.text[:600], "shape": "box",
                    "color": {"background": fill, "border": border}}
            if n.type == "evidence_span" and n.id not in cited:  # retrieved, unused
                node["color"] = {"background": "#f8fafc", "border": "#cbd5e1"}
                node["shapeProperties"] = {"borderDashes": [4, 4]}
                node["font"] = {"color": "#94a3b8"}
            vnodes.append(node)

        vedges = []
        for e in self.edges:
            if e.type == "answers":
                continue
            if e.type in _EDGE_FOR_VERDICT.values():  # evidence -> claim, meaningful
                border = _PALETTE[
                    {v: k for k, v in _EDGE_FOR_VERDICT.items()}[e.type]][1]
                vedges.append({"from": e.source, "to": e.target, "arrows": "to",
                               "label": e.type.replace("partially_supports", "partial")
                                              .replace("_", " "),
                               "font": {"size": 10, "color": border},
                               "color": {"color": border}, "width": 2})
            else:  # part_of_answer: quiet structural link
                vedges.append({"from": e.source, "to": e.target, "arrows": "to",
                               "color": {"color": "#cbd5e1"}, "width": 1})

        q = self.node("q")
        s = self.summary()
        stats = (f"{s['claims']} claims · support {s['claim_support_rate']} · "
                 f"weighted error {s['weighted_claim_error_rate']}"
                 if s["claims"] else "no claims (abstained)")
        return _HTML_TEMPLATE.format(
            title=title, question=(q.text if q else ""), stats=stats,
            nodes=json.dumps(vnodes), edges=json.dumps(vedges))

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


_HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
 body{{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;margin:0;color:#0f172a;background:#fff}}
 header{{padding:14px 22px;border-bottom:1px solid #e2e8f0}}
 h2{{margin:0;font-size:17px}} .q{{margin:4px 0 0;color:#475569;font-size:14px}}
 .stats{{margin-top:6px;font-size:12px;color:#334155;background:#f1f5f9;
        display:inline-block;padding:3px 10px;border-radius:12px}}
 #wrap{{position:relative}} #g{{width:100%;height:86vh}}
 #legend{{position:absolute;top:10px;right:14px;background:#ffffffe6;border:1px solid #e2e8f0;
         border-radius:8px;padding:8px 12px;font-size:11.5px;line-height:1.9;z-index:5}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
     border:1.5px solid}}
</style></head><body>
<header><h2>{title}</h2><p class="q">{question}</p><span class="stats">{stats}</span></header>
<div id="wrap"><div id="legend">
 <span class="sw" style="background:#dbeafe;border-color:#3b82f6"></span>evidence (cited)<br>
 <span class="sw" style="background:#f8fafc;border-color:#cbd5e1;border-style:dashed"></span>evidence (unused)<br>
 <span class="sw" style="background:#dcfce7;border-color:#16a34a"></span>claim · supported<br>
 <span class="sw" style="background:#fef3c7;border-color:#d97706"></span>claim · partial<br>
 <span class="sw" style="background:#fee2e2;border-color:#dc2626"></span>claim · unsupported<br>
 <span class="sw" style="background:#e0e7ff;border-color:#4f46e5"></span>conclusion</div>
<div id="g"></div></div><script>
const net = new vis.Network(document.getElementById("g"),
  {{nodes:new vis.DataSet({nodes}), edges:new vis.DataSet({edges})}},
  {{layout:{{hierarchical:{{direction:"LR",levelSeparation:280,nodeSpacing:120,
                        treeSpacing:60,sortMethod:"directed"}}}},
   physics:false,
   nodes:{{borderWidth:1.5,shapeProperties:{{borderRadius:6}},
          font:{{multi:"md",size:12.5,color:"#1e293b",face:"-apple-system, Segoe UI, Helvetica"}},
          widthConstraint:{{maximum:250}},margin:8}},
   edges:{{smooth:{{type:"cubicBezier",forceDirection:"horizontal",roundness:.45}},
          arrows:{{to:{{scaleFactor:.6}}}}}},
   interaction:{{hover:true,tooltipDelay:120,zoomSpeed:.6}}}});
net.once("afterDrawing",()=>net.fit({{animation:false}}));
</script></body></html>"""


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
