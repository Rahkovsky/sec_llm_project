#!/usr/bin/env python3
"""Incrementally update a Chroma index into a NEW directory (no full re-embed).

1. Clone the source index (APFS ``cp -c``: instant, copy-on-write).
2. Delete chunks of filings whose accession is no longer under the input dirs
   (e.g. 10-K/A amendments moved out by tools/refetch_10k.sh).
3. Embed + add every filing under the input dirs whose accession is not in the
   index yet, with metadata/sections from the download sidecar (the same path as
   IndexBuilder.build_from_dir, so no tools/repair_metadata.py pass is needed).
4. ``--gold CASES``: also add each FinDER gold passage as its own pseudo-filing
   (filing_type 10-K, accession FINDER-<id>, source_url finder_gold) -> the
   "oracle corpus" condition, where the evidence is guaranteed to be present.
5. finalize_index: rebuild chunks.jsonl, index_meta.json and bm25.json.

    python tools/update_index.py --config config/local_16gb.yaml \
        --src data/sec_index_qwen06 --dst data/sec_index_v3 \
        data/input/10-K data/input/10-Q data/input/8-K
    python tools/update_index.py --config config/local_16gb.yaml \
        --src data/sec_index_v3 --dst data/sec_index_v3_gold --gold eval/finder_semantic.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sec_llm_project.sec_intel.chunking.sec_sections import (  # noqa: E402
    metadata_from_filename,
    sections_from_sidecar,
)
from sec_llm_project.sec_intel.core.config import AppConfig  # noqa: E402
from sec_llm_project.sec_intel.core.types import FilingMetadata  # noqa: E402
from sec_llm_project.sec_intel.index.builder import (  # noqa: E402
    IndexBuilder,
    _infer_filing_type,
    _load_sidecar,
)
from sec_llm_project.sec_intel.index.finalize import finalize_index  # noqa: E402
from sec_llm_project.sec_intel.index.store import SECIndex  # noqa: E402


def file_chunks(fp: Path, chunker):
    text = fp.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None, []
    sc = _load_sidecar(fp)
    if sc:
        md = FilingMetadata.from_dict(sc["metadata"])
        return md, chunker.chunk_filing(text, md, sections=sections_from_sidecar(sc, text) or None)
    md = metadata_from_filename(fp, filing_type=_infer_filing_type(fp, "10-K"))
    return md, chunker.chunk_filing(text, md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dirs", nargs="*")
    ap.add_argument("--config", default="config/local_16gb.yaml")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--gold", help="FinDER cases JSON: add gold passages (oracle corpus)")
    ap.add_argument("--flush", type=int, default=256, help="chunks per embed/insert batch")
    ap.add_argument("--no-delete", action="store_true")
    ap.add_argument("--dtype", help="embedder dtype for documents (float16 is ~1.5x faster than "
                    "bfloat16 on an M1 Pro; cos to bf16 vectors >= 0.9998)")
    ap.add_argument("--tickers-from", action="append", default=[],
                    help="cases JSON: embed new filings only for these tickers (repeatable); "
                         "re-run without it to add the rest")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not dst.exists():
        print(f"cloning {src} -> {dst}", flush=True)
        subprocess.run(["cp", "-cR", str(src), str(dst)], check=True)
    cfg = AppConfig.from_file(args.config)
    cfg.index.path = str(dst)
    if args.dtype:
        cfg.embedding.dtype = args.dtype
    meta = json.loads((dst / "index_meta.json").read_text())

    # accession -> (ticker, form) currently in the index; company name per ticker
    have: dict[str, str] = {}
    company: dict[str, str] = {}
    with open(dst / "chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)["metadata"]
            have.setdefault(m["accession"], m.get("ticker", ""))
            if m.get("company", "UNKNOWN") != "UNKNOWN":
                company.setdefault(m["ticker"], m["company"])
    print(f"index: {meta['chunk_count']:,} chunks, {len(have):,} filings", flush=True)

    builder = IndexBuilder(cfg)
    files = [fp for d in args.input_dirs for fp in sorted(Path(d).rglob("*.txt")) if fp.is_file()]
    on_disk: dict[str, Path] = {}
    for fp in files:
        sc = _load_sidecar(fp)
        acc = (sc or {}).get("metadata", {}).get("accession") or \
            metadata_from_filename(fp).accession
        on_disk[acc] = fp

    index = SECIndex(builder.embedder.info, store="chroma", path=str(dst),
                     collection=cfg.index.collection)
    col = index._chroma_collection

    if args.input_dirs and not args.no_delete:
        gone = [a for a in have if a not in on_disk and not a.startswith("FINDER-")]
        print(f"deleting {len(gone)} filings no longer on disk: {gone[:8]}...", flush=True)
        for a in gone:
            col.delete(where={"accession": a})

    todo = [(a, fp) for a, fp in on_disk.items() if a not in have]
    if args.tickers_from:
        keep = {c["ticker"] for f in args.tickers_from for c in json.loads(Path(f).read_text())}
        skipped = [a for a, fp in todo if fp.parent.name not in keep]
        todo = [(a, fp) for a, fp in todo if fp.parent.name in keep]
        print(f"--tickers-from: {len(todo)} new filings for {len(keep)} tickers; "
              f"{len(skipped)} left for a later run", flush=True)
    # Resume: chunks.jsonl is only rewritten by finalize_index at the end, so filings
    # embedded by an interrupted run are in Chroma but not in `have`. Re-chunk each
    # pending filing (cheap, no embedding) and skip it when Chroma already holds all
    # of its chunks; a partial filing (killed mid-insert) is deleted and redone.
    resumed, pending = 0, []
    for a, fp in todo:
        n_in = len(col.get(where={"accession": a}, include=[])["ids"])
        if n_in:
            _md, chunks = file_chunks(fp, builder.chunker)
            if n_in == len(chunks):
                resumed += 1
                continue
            col.delete(where={"accession": a})
        pending.append((a, fp))
    if resumed:
        print(f"resume: {resumed} filings already embedded by an earlier run", flush=True)
    todo = pending
    print(f"new filings to embed: {len(todo)} "
          f"{dict(Counter(_infer_filing_type(fp, '10-K') for _, fp in todo))}", flush=True)
    buf, done, t0 = [], 0, time.time()

    def flush() -> None:
        nonlocal buf, done
        if buf:
            index.add(buf, builder.embedder.embed_documents([c.text for c in buf]))
            done += len(buf)
            buf = []
            rate = done / (time.time() - t0)
            print(f"[{time.strftime('%H:%M:%S')}] embedded {done:,} chunks ({rate:.1f}/s)",
                  flush=True)

    for _a, fp in todo:
        _md, chunks = file_chunks(fp, builder.chunker)
        buf.extend(chunks)
        if len(buf) >= args.flush:
            flush()
    flush()

    if args.gold:
        cases = json.loads(Path(args.gold).read_text())
        n0 = done
        for c in cases:
            for i, p in enumerate(c["relevant_passages"]):
                md = FilingMetadata(ticker=c["ticker"], company=company.get(c["ticker"], "UNKNOWN"),
                                    filing_type="10-K", filing_date="UNKNOWN",
                                    fiscal_year="UNKNOWN", accession=f"FINDER-{c['id']}-{i}",
                                    source_url="finder_gold", source_path=f"finder_gold/{c['id']}-{i}")
                buf.extend(builder.chunker.chunk_filing(p, md))
        flush()
        print(f"gold: added {done - n0} chunks from {len(cases)} questions", flush=True)

    total = finalize_index(dst, collection=cfg.index.collection,
                           embedding_info=meta["embedding_info"])
    print(f"DONE {dst}: {total:,} chunks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
