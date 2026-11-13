#!/usr/bin/env python3
"""Run gold questions through the pipeline and emit claim-provenance DAGs.

For each question: answer with citation verification on, build the DAG, and
write <out>/<qid>/ {answer.json, dag.json, dag.dot, mistakes.csv} plus a
combined summary.csv. Renders dag.png too when the `dot` binary is installed.

    python tools/provenance_dag.py --config config/local.yaml -n 8
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.llm import providers
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline
from sec_llm_project.sec_intel.provenance import build_dag
from sec_llm_project.utils.env_config import load_environment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--cases", default="eval/finder_semantic.json")
    ap.add_argument("-n", type=int, default=8, help="number of gold questions")
    ap.add_argument("--out", default="out/provenance")
    ap.add_argument("--entity", choices=["case", "resolved"], default="case",
                    help="ticker filter: the case's own ticker (oracle) or resolved from the "
                         "question text (tools/eval_entity.py measures the resolver)")
    ap.add_argument("--workers", type=int, default=1,
                    help="questions in flight; >1 groups LLM calls by model (fewer model "
                         "swaps on a one-model-at-a-time machine), same calls per answer")
    args = ap.parse_args()

    load_environment()  # .env (HF_HUB_OFFLINE etc.), same as the CLI does
    cfg = AppConfig.load(args.config)
    cfg.verification.enabled = True  # claim verdicts are the DAG's edges
    pipeline = SECIntelPipeline(cfg)
    pipeline.load_index()

    cases = json.loads(Path(args.cases).read_text())[: args.n]
    dot = shutil.which("dot")
    fields = ["id", "ticker", "claims", "claim_support_rate", "weighted_claim_error_rate",
              "missing_evidence_claims", "contradicted_claims", "rejected_conclusion", "error"]
    summary_path = Path(args.out) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary_path.exists()

    def run_case(case: dict, w: csv.DictWriter, sfh) -> bool:
        qid, question, ticker = case["id"], case["question"], case.get("ticker")
        d = Path(args.out) / qid
        print(f"[{qid}] {question[:80]}", flush=True)
        providers.reset_call_log()
        t_q = time.perf_counter()
        try:
            if args.entity == "resolved":
                filters, res = pipeline.resolve_filters(question)
                d.mkdir(parents=True, exist_ok=True)
                (d / "entity.json").write_text(json.dumps(
                    {"gold": ticker, "resolved": res.tickers, "evidence": res.evidence}))
            else:
                filters = {"ticker": ticker} if ticker else None
            answer = pipeline.ask(question, filters=filters)
            dag = build_dag(answer.to_dict())
        except Exception as exc:  # keep an overnight batch alive; row records it
            print(f"    [{qid}] ERROR: {exc}", flush=True)
            with lock:
                w.writerow({"id": qid, "ticker": ticker, "error": str(exc)[:200]})
                sfh.flush()
            return False

        d.mkdir(parents=True, exist_ok=True)
        (d / "timings.json").write_text(json.dumps(
            {"total_s": round(time.perf_counter() - t_q, 3), "workers": args.workers,
             "calls": providers.call_log()}, indent=1))
        (d / "answer.json").write_text(answer.to_json())
        (d / "dag.json").write_text(dag.to_json())
        (d / "dag.dot").write_text(dag.to_dot())
        (d / "dag.html").write_text(dag.to_html(qid))
        if dot:
            subprocess.run([dot, "-Tpng", "-Gdpi=170", str(d / "dag.dot"),
                            "-o", str(d / "dag.png")], check=False)
            subprocess.run([dot, "-Tpdf", str(d / "dag.dot"),
                            "-o", str(d / "dag.pdf")], check=False)  # vector, for slides
        rows = dag.mistake_table()
        if rows:
            with open(d / "mistakes.csv", "w", newline="") as fh:
                mw = csv.DictWriter(fh, fieldnames=rows[0].keys())
                mw.writeheader()
                mw.writerows(rows)

        s = dag.summary()
        with lock:
            w.writerow({"id": qid, "ticker": ticker, **s})
            sfh.flush()  # summary survives a crash at any point
        print(f"    [{qid}] claims={s['claims']} support_rate={s['claim_support_rate']} "
              f"weighted_error={s['weighted_claim_error_rate']} "
              f"rejected={s['rejected_conclusion']}", flush=True)
        return True

    todo = [c for c in cases if not (Path(args.out) / c["id"] / "dag.json").exists()]
    lock = threading.Lock()
    with open(summary_path, "a", newline="") as sfh:
        w = csv.DictWriter(sfh, fieldnames=fields)
        if new_file:
            w.writeheader()
        if args.workers <= 1:
            done = sum(run_case(c, w, sfh) for c in todo)
        else:
            # Questions in parallel threads; the scheduler groups their LLM calls by
            # model so the generator and judge are swapped rarely. Shared state is
            # made safe here: BM25 and the retriever load once up front, and the
            # embedder (torch on one GPU) runs one call at a time.
            retriever = pipeline.retriever
            retriever._bm25_index()
            emb = pipeline.embedder
            emb_lock = threading.Lock()
            for name in ("embed_query", "embed_documents"):
                fn = getattr(emb, name)

                def locked(*a, _fn=fn, **k):
                    with emb_lock:
                        return _fn(*a, **k)
                setattr(emb, name, locked)
            sched = providers.SCHEDULER = providers.ModelScheduler()

            def job(c: dict) -> bool:
                with sched.worker():
                    return run_case(c, w, sfh)

            with ThreadPoolExecutor(args.workers) as pool:
                done = sum(pool.map(job, todo))
            print(f"model switches: {sched.switches} for {len(todo)} questions", flush=True)

    print(f"\nWrote {done} DAGs to {args.out}/ (summary: {summary_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
