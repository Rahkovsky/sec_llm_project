#!/usr/bin/env python3
"""Synthetic, Claude-written Q/A over random index passages (retrieval + RAG eval).

Why: FinDER queries are terse evaluative notes, and only 162/260 of its gold
passages are even findable under the strict rule. A model-independent set of
well-formed questions with known answers and verbatim evidence separates
"the embedder is weak" from "the benchmark's query style is unusual". Claude
writes the queries so neither embedder family (Qwen / Nomic) is favoured by a
same-family generator.

Per passage Claude returns: usable flag, natural question, the same need in
FinDER's terse style, a short answer, and a VERBATIM evidence quote. Relevance
for retrieval = every chunk of that ticker containing the evidence (10-Ks repeat
text across years, so the source chunk alone would punish retrieving an
identical copy). The same batch rewrites the FinDER dev queries of those tickers
as clean questions (hypothesis H1 in tools/ab_diagnose.py).

    python tools/make_synth_qa.py --pilot 3     # 3 synchronous calls, print them
    python tools/make_synth_qa.py --submit      # Batches API (50% price)
    python tools/make_synth_qa.py --collect     # poll, then write outputs

Outputs: out/synth_qa/{sample,results}.json, out/ab_diag/queries.json (for
ab_diagnose), eval/synth_claude.json (FinDER-format eval set).
Credentials: ANTHROPIC_API_KEY, else the file ~/.anthropic_key (one line).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

INDEX = Path("data/sec_index_qwen06")
CACHE = Path("out/ab_embedders/nomic")
OUT = Path("out/synth_qa")
MODEL = "claude-fable-5-2"  # top Claude model: synthetic eval data is never written by local models

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "reason_if_unusable": {"type": "string"},
        "question": {"type": "string"},
        "terse": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
        "answer_type": {"type": "string", "enum": ["numeric", "factual", "descriptive"]},
    },
    "required": ["usable", "reason_if_unusable", "question", "terse", "answer", "evidence",
                 "answer_type"],
    "additionalProperties": False,
}
REWRITE_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}},
                  "required": ["question"], "additionalProperties": False}

SYSTEM = """You build a retrieval benchmark over SEC filings (10-K, 10-Q, 8-K).
You receive ONE passage from one company's filing. Write an information need that
THIS passage satisfies.

Rules:
- usable=false if the passage is a table of contents, signature block, exhibit
  list, cover-page boilerplate, or contains no company-specific fact (e.g. generic
  risk language any company could file). Then fill the other fields with "".
- question: one natural, specific question a financial analyst would ask, that
  this passage answers. Name the company (or ticker). Paraphrase: never copy more
  than 5 consecutive words from the passage. Do not mention "the passage" or "the
  filing section".
- terse: the SAME information need, rewritten in the style of these analyst
  notes (statement form, abbreviations, ticker, some evaluative filler):
{style}
- answer: 1-3 sentences, fully supported by the passage.
- evidence: an EXACT, character-for-character copy of the 1-3 consecutive
  sentences of the passage that support the answer (no ellipses, no edits).
- answer_type: numeric (a figure/date/amount), factual (a name/yes-no/list),
  or descriptive (an explanation/policy/process)."""


def load_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        f = Path.home() / ".anthropic_key"
        if f.exists():
            os.environ["ANTHROPIC_API_KEY"] = f.read_text().strip().removeprefix(
                "export ANTHROPIC_API_KEY=").strip("'\"")


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def usable_chunk(tx: str) -> bool:
    if len(tx) < 800:
        return False
    letters = sum(c.isalpha() for c in tx)
    digits = sum(c.isdigit() for c in tx)
    return letters / len(tx) > 0.55 and digits / len(tx) < 0.2


def sample(n: int, seed: int, n_tickers: int = 40) -> list[dict]:
    # Same 40 tickers as `ab_embedders.py --split dev --max-tickers 40` (Nomic-cached).
    dev = set(json.loads(Path("eval/finder_split.json").read_text())["dev"])
    tickers = sorted({c["ticker"] for c in json.loads(Path("eval/finder_semantic.json")
                      .read_text()) if c["id"] in dev})[:n_tickers]
    want = set(tickers)
    pool: dict[str, list[dict]] = defaultdict(list)
    with open(INDEX / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            md = d["metadata"]
            if md.get("ticker") in want and usable_chunk(d["text"]):
                pool[md["ticker"]].append({
                    "chunk_id": d["chunk_id"], "ticker": md["ticker"], "text": d["text"],
                    **{k: md.get(k) for k in ("filing_type", "filing_date", "item_number",
                                              "section_title")}})
    rng = random.Random(seed)
    per, extra = divmod(n, len(tickers))
    picks = []
    for i, t in enumerate(tickers):
        picks += rng.sample(pool[t], min(len(pool[t]), per + (i < extra)))
    rng.shuffle(picks)
    for i, p in enumerate(picks):
        p["sid"] = f"s{i:04d}"
    return picks


def finder_style(k: int = 8) -> str:
    cases = json.loads(Path("eval/finder_semantic.json").read_text())
    rng = random.Random(1)
    return "\n".join(f"  - {c['question']}" for c in rng.sample(cases, k))


def requests(picks: list[dict]) -> list[dict]:
    system = [{"type": "text", "text": SYSTEM.format(style=finder_style()),
               "cache_control": {"type": "ephemeral"}}]
    reqs = []
    for p in picks:
        user = (f"Company ticker: {p['ticker']}\nFiling: {p['filing_type']} filed "
                f"{p['filing_date']}, item {p['item_number']} ({p['section_title']})\n\n"
                f"<passage>\n{p['text']}\n</passage>")
        reqs.append({"custom_id": p["sid"], "params": {
            "model": MODEL, "max_tokens": 16000, "system": system,  # thinking counts
            "output_config": {"effort": "medium",
                              "format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": user}]}})
    tick = {p["ticker"] for p in picks}
    dev = set(json.loads(Path("eval/finder_split.json").read_text())["dev"])
    for c in json.loads(Path("eval/finder_semantic.json").read_text()):
        if c["ticker"] in tick and c["id"] in dev:
            reqs.append({"custom_id": "fq_" + c["id"], "params": {
                "model": MODEL, "max_tokens": 8000,
                "output_config": {"effort": "low",
                                  "format": {"type": "json_schema", "schema": REWRITE_SCHEMA}},
                "messages": [{"role": "user", "content":
                              "Rewrite this analyst note as one clear, neutral question about "
                              "what the company's SEC filing discloses. Drop opinions and "
                              "filler; keep the topic, the company name and ticker.\n"
                              f"Note: {c['question']}"}]}})
    return reqs


def text_of(msg) -> str:  # noqa: ANN001 - SDK Message
    return next((b.text for b in msg.content if b.type == "text"), "")


def write_outputs(picks: list[dict], results: dict[str, dict]) -> None:
    by_t: dict[str, list[tuple[str, str]]] = defaultdict(list)
    tick = {p["ticker"] for p in picks}
    with open(INDEX / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["metadata"].get("ticker") in tick:
                by_t[d["metadata"]["ticker"]].append((d["chunk_id"], norm_ws(d["text"])))
    synth, evalset, stats = [], [], defaultdict(int)
    for p in picks:
        r = results.get(p["sid"])
        if r is None:
            stats["missing"] += 1
            continue
        if not r["usable"]:
            stats["unusable"] += 1
            continue
        ev = norm_ws(r["evidence"])
        if len(ev) < 30 or ev not in norm_ws(p["text"]):
            stats["evidence_not_verbatim"] += 1
            continue
        rel = [cid for cid, tx in by_t[p["ticker"]] if ev in tx]
        stats["kept"] += 1
        stats["rel_ids_total"] += len(rel)
        synth.append({"ticker": p["ticker"], "chunk_id": p["chunk_id"], "rel_ids": rel,
                      "question": r["question"], "terse": r["terse"], "answer": r["answer"],
                      "evidence": r["evidence"], "answer_type": r["answer_type"],
                      "item_number": p["item_number"], "chunk_len": len(p["text"])})
        evalset.append({"id": f"synth_{p['sid']}", "question": r["question"],
                        "ticker": p["ticker"], "category": f"item_{p['item_number']}",
                        "relevant_passages": [r["evidence"]],
                        "relevant_items": [p["item_number"]], "answerable": True,
                        "reference_answer": r["answer"], "answer_type": r["answer_type"],
                        "source": f"claude:{MODEL} from {p['chunk_id']}"})
    finder_q = {k[3:]: v["question"] for k, v in results.items() if k.startswith("fq_")}
    Path("out/ab_diag").mkdir(parents=True, exist_ok=True)
    Path("out/ab_diag/queries.json").write_text(json.dumps(
        {"synth": synth, "finder_q": finder_q, "generator": MODEL}, indent=1))
    Path("eval/synth_claude.json").write_text(json.dumps(evalset, indent=1))
    print(dict(stats), f"| finder rewrites: {len(finder_q)}")
    print(f"mean relevant chunks per kept question: "
          f"{stats['rel_ids_total'] / max(stats['kept'], 1):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pilot", type=int, default=0, help="N synchronous calls, printed")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    load_key()
    import anthropic
    client = anthropic.Anthropic()
    OUT.mkdir(parents=True, exist_ok=True)
    sample_path = OUT / "sample.json"
    if sample_path.exists():
        picks = json.loads(sample_path.read_text())
    else:
        picks = sample(args.n, args.seed)
        sample_path.write_text(json.dumps(picks, indent=1))
    print(f"{len(picks)} passages over {len({p['ticker'] for p in picks})} tickers")
    reqs = requests(picks)

    if args.pilot:
        for rq in reqs[: args.pilot]:
            msg = client.messages.create(**rq["params"])
            print(f"\n[{rq['custom_id']}] stop={msg.stop_reason} in={msg.usage.input_tokens} "
                  f"out={msg.usage.output_tokens}")
            print(json.dumps(json.loads(text_of(msg)), indent=1))
        return 0
    if args.submit:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = client.messages.batches.create(requests=[
            Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(
                **r["params"])) for r in reqs])
        (OUT / "batch.json").write_text(json.dumps({"id": batch.id, "n": len(reqs)}))
        print(f"submitted batch {batch.id} with {len(reqs)} requests")
        return 0
    if args.collect:
        bid = json.loads((OUT / "batch.json").read_text())["id"]
        while (b := client.messages.batches.retrieve(bid)).processing_status != "ended":
            print(f"  {b.processing_status}: {b.request_counts.processing} processing, "
                  f"{b.request_counts.succeeded} done", flush=True)
            time.sleep(60)
        results, bad, usage = {}, defaultdict(int), defaultdict(int)
        for res in client.messages.batches.results(bid):
            if res.result.type != "succeeded":
                bad[res.result.type] += 1
                continue
            msg = res.result.message
            if msg.stop_reason != "end_turn":
                bad[f"stop_{msg.stop_reason}"] += 1
                continue
            usage["in"] += msg.usage.input_tokens
            usage["out"] += msg.usage.output_tokens
            usage["cache_read"] += msg.usage.cache_read_input_tokens or 0
            results[res.custom_id] = json.loads(text_of(msg))
        (OUT / "results.json").write_text(json.dumps(results, indent=1))
        print(f"results: {len(results)} ok, problems: {dict(bad)}, usage: {dict(usage)}")
        write_outputs(picks, results)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
