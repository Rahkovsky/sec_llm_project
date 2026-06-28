"""Snapshot regression for evaluation metrics (report-only; never gates CI).

A run's aggregate metrics are frozen to a versioned JSON baseline so a later run
can show per-metric deltas. A drop in a higher-is-better metric (or a rise in a
lower-is-better one) beyond a tolerance is flagged as a regression for human
review — it deliberately does NOT change the process exit code, so a noisy
baseline can never block a merge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HIGHER_IS_BETTER = {
    "recall@k", "mrr", "map@k", "citation_correctness", "abstention_recall",
    "mean_groundedness", "fully_grounded_rate",
}
LOWER_IS_BETTER = {
    "hallucination_rate", "contradiction_rate", "mean_latency_ms", "p95_latency_ms",
}


def save_baseline(report: dict[str, Any], path: str | Path) -> None:
    """Freeze the aggregate metrics (latency excluded — it is environment-dependent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    metrics = {k: v for k, v in report.get("metrics", {}).items() if not k.endswith("_latency_ms")}
    payload = {"k": report.get("k"), "n_cases": report.get("n_cases"), "metrics": metrics}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def diff_metrics(current: dict[str, Any], baseline: dict[str, Any], *,
                 tol: float = 0.01) -> dict[str, Any]:
    """Per-metric deltas + a list of regressions (beyond ``tol``).

    Latency is environment-dependent, so it is excluded from the comparison
    (consistent with :func:`save_baseline`).
    """
    cur = {k: v for k, v in current.get("metrics", {}).items() if not k.endswith("_latency_ms")}
    base = {k: v for k, v in baseline.get("metrics", {}).items() if not k.endswith("_latency_ms")}
    deltas: dict[str, dict[str, float]] = {}
    regressions: list[str] = []
    for name in sorted(set(cur) & set(base)):
        c, b = float(cur[name]), float(base[name])
        delta = c - b
        deltas[name] = {"baseline": round(b, 4), "current": round(c, 4), "delta": round(delta, 4)}
        if (name in HIGHER_IS_BETTER and delta < -tol) or (name in LOWER_IS_BETTER and delta > tol):
            regressions.append(name)
    return {
        "deltas": deltas,
        "regressions": regressions,
        "added": sorted(set(cur) - set(base)),
        "removed": sorted(set(base) - set(cur)),
    }


def render_diff(diff: dict[str, Any]) -> str:
    lines = ["", "Snapshot vs baseline (report-only, no CI gate):", "-" * 46]
    for name, d in diff["deltas"].items():
        flag = "   <== REGRESSION" if name in diff["regressions"] else ""
        lines.append(f"  {name:24s}: {d['baseline']:.4f} -> {d['current']:.4f} "
                     f"({d['delta']:+.4f}){flag}")
    if diff["added"]:
        lines.append(f"  (new, not in baseline: {', '.join(diff['added'])})")
    lines.append("Regressions: " + (", ".join(diff["regressions"]) if diff["regressions"]
                                     else "none beyond tolerance"))
    return "\n".join(lines)
