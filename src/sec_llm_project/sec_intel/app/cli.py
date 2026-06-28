"""Command-line demo for the SEC Disclosure Intelligence Prototype (plan item 12).

Subcommands:

* ``fetch``    — download official EDGAR filings (multi-form, multi-year; --sp500 for full index)
* ``index``    — build an index from a directory of filings
* ``ask``      — grounded Q&A with citations, confidence, and abstention
* ``extract``  — structured JSON extraction (risk_factors, litigation, ...)
* ``compare``  — diff two filing years for a company
* ``monitor``  — Disclosure Change & Risk Signal Monitor (flagship)
* ``evaluate`` — run the curated benchmark
* ``demo``     — index the built-in synthetic corpus and answer a sample question

Backends are chosen via ``--config`` (JSON/YAML) and/or ``SECI_*`` env vars, so
no code changes are needed to switch from the offline mock to Ollama/OpenAI/
Anthropic and from hashing to neural embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..core.config import AppConfig
from ..pipeline import SECIntelPipeline


def _load_config(path: str | None) -> AppConfig:
    return AppConfig.load(path)


def _print_answer(answer) -> None:
    print(f"\nQ: {answer.question}")
    flag = " [ABSTAINED]" if answer.abstained else ""
    print(f"Confidence: {answer.confidence:.2f}{flag}  (model: {answer.model})")
    if answer.abstained:
        print(f"Reason: {answer.abstain_reason}")
    print(f"\n{answer.text}\n")
    if answer.citations:
        print("Sources:")
        for i, c in enumerate(answer.citations, 1):
            src = c.source_url or c.chunk_id
            print(f"  [{i}] {c.company} {c.filing_type} {c.filing_date} — "
                  f"{c.section_title} (Item {c.item_number}), score={c.score}")
            print(f"      \"{c.quote[:160]}\"  <{src}>")


def _parse_years(spec: list[str] | None) -> list[int]:
    """Expand ``--years`` tokens; supports ranges like ``2021-2024``."""
    years: set[int] = set()
    for token in spec or []:
        if "-" in token:
            lo, hi = token.split("-", 1)
            years.update(range(int(lo), int(hi) + 1))
        else:
            years.add(int(token))
    return sorted(years)


def _resolve_fetch_tickers(args: argparse.Namespace) -> list[str]:
    """Merge explicit tickers + optional S&P 500 universe."""
    tickers: set[str] = {t.upper() for t in (args.tickers or [])}
    if getattr(args, "sp500", False) or getattr(args, "sp500_live", False):
        from ..ingest.sp500 import get_sp500_tickers
        live = getattr(args, "sp500_live", False)
        if live:
            print("Fetching current S&P 500 list from Wikipedia...", flush=True)
        sp500 = get_sp500_tickers(live=live)
        tickers.update(sp500)
        label = "live" if live else "bundled Q1-2025"
        print(f"Universe: S&P 500 ({label}, {len(sp500)} symbols) "
              f"+ {len(args.tickers or [])} explicit → {len(tickers)} total")
    if not tickers:
        raise SystemExit("error: provide at least one ticker or use --sp500")
    return sorted(tickers)


def _cmd_fetch(args: argparse.Namespace) -> int:
    from ..ingest import download_filings

    tickers = _resolve_fetch_tickers(args)
    years = _parse_years(args.years)
    counts = download_filings(
        tickers, forms=args.forms, years=years or None,
        rate_per_sec=args.rate_per_sec, reset=args.reset,
    )
    total = sum(counts.values())
    span = f"{years[0]}-{years[-1]}" if years else "latest"
    print(f"Downloaded {total} filing(s) for {len(tickers)} ticker(s) ({span}):")
    for form, n in counts.items():
        print(f"  {form}: {n}")
    print("Next: sec-intel index data/input/10-K data/input/10-Q data/input/8-K")
    return 0 if total else 1


def _cmd_index(args: argparse.Namespace) -> int:
    pipeline = SECIntelPipeline(_load_config(args.config))
    dirs = args.input_dirs
    index = pipeline.build_index(
        dirs if len(dirs) > 1 else dirs[0],
        pattern=args.pattern, default_filing_type=args.filing_type,
        max_files=args.max_files,
    )
    label = ", ".join(dirs) if len(dirs) > 1 else dirs[0]
    print(f"Indexed {len(index)} chunks from {label} -> {pipeline.config.index.path}")
    print(f"Embedding: {pipeline.embedder.info.fingerprint()}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    pipeline = SECIntelPipeline(_load_config(args.config))
    pipeline.load_index()
    filters = {"ticker": args.ticker.upper()} if args.ticker else None
    answer = pipeline.ask(args.question, filters=filters, top_k=args.top_k)
    if args.json:
        print(answer.to_json())
    else:
        _print_answer(answer)
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    pipeline = SECIntelPipeline(_load_config(args.config))
    pipeline.load_index()
    result = pipeline.extract(args.target, ticker=args.ticker, top_k=args.top_k)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.valid else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    pipeline = SECIntelPipeline(_load_config(args.config))
    pipeline.load_index()
    report = pipeline.compare_years(
        args.ticker, args.year_a, args.year_b, filing_type=args.filing_type,
        items=args.items,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    pipeline = SECIntelPipeline(_load_config(args.config))
    if args.demo:
        from ..evaluation.dataset import index_sample_corpus

        index_sample_corpus(pipeline)
    else:
        pipeline.load_index()
    report = pipeline.monitor(
        args.ticker, args.year_a, args.year_b, forms=args.forms,
        compare_form=args.compare_form, xbrl=args.xbrl,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from ..evaluation.dataset import CURATED_CASES, index_sample_corpus, load_cases_from_json
    from ..evaluation.enforcement import load_enforcement_cases, run_enforcement_benchmark
    from ..evaluation.runner import EvaluationRunner

    pipeline = SECIntelPipeline(_load_config(args.config))
    if args.load_index:
        pipeline.load_index()  # evaluate against the persisted (e.g. full) index
    elif args.demo or not args.input_dir:
        index_sample_corpus(pipeline)
    else:
        pipeline.build_index(args.input_dir, persist=False)

    if args.enforcement:
        cases = load_enforcement_cases(args.cases) if args.cases else None
        report = run_enforcement_benchmark(pipeline, cases, k=args.top_k)
    else:
        cases = load_cases_from_json(args.cases) if args.cases else CURATED_CASES
        report = EvaluationRunner(pipeline).run(cases, k=args.top_k, generate=not args.no_generate)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    # Snapshot regression (report-only; never changes the exit code).
    if not args.enforcement and (args.snapshot or args.update_baseline):
        from ..evaluation.snapshot import diff_metrics, load_baseline, render_diff, save_baseline

        rep = report.to_dict()
        if args.update_baseline:
            save_baseline(rep, args.update_baseline)
            print(f"\nBaseline written: {args.update_baseline}")
        if args.snapshot:
            baseline = load_baseline(args.snapshot)
            if baseline is None:
                print(f"\nNo baseline at {args.snapshot} "
                      f"(create one with: --update-baseline {args.snapshot})")
            else:
                print(render_diff(diff_metrics(rep, baseline)))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from ..evaluation.dataset import index_sample_corpus

    pipeline = SECIntelPipeline(_load_config(args.config))
    index_sample_corpus(pipeline)
    print(f"Indexed built-in synthetic corpus ({len(pipeline.index)} chunks).")
    question = args.question or "What supply chain risks does Nova Devices face?"
    _print_answer(pipeline.ask(question, filters={"ticker": "NOVA"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sec-intel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to JSON/YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Download official EDGAR filings (needs .[sec])")
    p_fetch.add_argument("tickers", nargs="*", default=[],
                         help="Ticker symbols (optional when --sp500 is set)")
    p_fetch.add_argument("--sp500", action="store_true",
                         help="Include all S&P 500 companies (bundled Q1-2025 list)")
    p_fetch.add_argument("--sp500-live", action="store_true",
                         help="Like --sp500 but fetches the current list from Wikipedia")
    p_fetch.add_argument("--forms", nargs="*", default=["10-K", "10-Q", "8-K"],
                         help="Form types (default: 10-K 10-Q 8-K; also DEF 14A)")
    p_fetch.add_argument("--years", nargs="*", default=None,
                         help="Years or ranges, e.g. 2023 2024 or 2021-2024 "
                              "(omit for latest only)")
    p_fetch.add_argument("--rate-per-sec", type=float, default=8.0,
                         help="EDGAR fair-access rate cap (<= 10 req/s)")
    p_fetch.add_argument("--reset", action="store_true",
                         help="Clear the checkpoint and start fresh (re-download everything)")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_index = sub.add_parser("index", help="Build an index from one or more filing directories")
    p_index.add_argument("input_dirs", nargs="+", metavar="DIR",
                         help="One or more form directories (e.g. data/input/10-K data/input/10-Q)")
    p_index.add_argument("--pattern", default="*.txt")
    p_index.add_argument("--filing-type", default="10-K")
    p_index.add_argument("--max-files", type=int, default=None)
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a grounded question")
    p_ask.add_argument("question")
    p_ask.add_argument("--ticker", default=None)
    p_ask.add_argument("--top-k", type=int, default=8)
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=_cmd_ask)

    p_ext = sub.add_parser("extract", help="Structured extraction")
    p_ext.add_argument("target", help="risk_factors|litigation|mdna|internal_controls|"
                                      "related_party|liquidity")
    p_ext.add_argument("--ticker", default=None)
    p_ext.add_argument("--top-k", type=int, default=8)
    p_ext.set_defaults(func=_cmd_extract)

    p_cmp = sub.add_parser("compare", help="Compare two filing years")
    p_cmp.add_argument("ticker")
    p_cmp.add_argument("year_a")
    p_cmp.add_argument("year_b")
    p_cmp.add_argument("--filing-type", default="10-K")
    p_cmp.add_argument("--items", nargs="*", default=None)
    p_cmp.set_defaults(func=_cmd_compare)

    p_mon = sub.add_parser("monitor", help="Disclosure Change & Risk Signal Monitor")
    p_mon.add_argument("ticker")
    p_mon.add_argument("year_a")
    p_mon.add_argument("year_b")
    p_mon.add_argument("--forms", nargs="*", default=["10-K", "10-Q", "8-K"])
    p_mon.add_argument("--compare-form", default="10-K")
    p_mon.add_argument("--xbrl", action="store_true", help="Attach XBRL context (needs network)")
    p_mon.add_argument("--demo", action="store_true", help="Use built-in synthetic corpus")
    p_mon.set_defaults(func=_cmd_monitor)

    p_eval = sub.add_parser("evaluate", help="Run the curated benchmark")
    p_eval.add_argument("--input-dir", default=None, help="Corpus dir (default: synthetic)")
    p_eval.add_argument("--load-index", action="store_true",
                        help="Evaluate against the persisted index (config index.path) "
                             "instead of rebuilding — use with the full local index")
    p_eval.add_argument("--cases", default=None, help="JSON file of eval cases")
    p_eval.add_argument("--demo", action="store_true", help="Use built-in synthetic corpus")
    p_eval.add_argument("--enforcement", action="store_true",
                        help="Run the enforcement-case benchmark (evaluation, not training)")
    p_eval.add_argument("--top-k", type=int, default=8)
    p_eval.add_argument("--no-generate", action="store_true", help="Retrieval metrics only")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--snapshot", nargs="?", const="tests/eval/baseline.json", default=None,
                        help="Compare metrics to a baseline JSON and print deltas (no CI gate)")
    p_eval.add_argument("--update-baseline", nargs="?", const="tests/eval/baseline.json",
                        default=None, help="Write the current run's metrics as the baseline JSON")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_demo = sub.add_parser("demo", help="Run an offline end-to-end demo")
    p_demo.add_argument("--question", default=None)
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
