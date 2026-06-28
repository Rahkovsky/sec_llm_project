#!/usr/bin/env python3
"""Optimized bulk indexer for the local stack (Nomic + Chroma).

Why this exists
---------------
The default per-file indexer embeds one filing at a time and inserts to Chroma
inline, so the GPU sits idle during inserts and vice-versa. Profiling on an
M1 Pro showed the embedding model (Nomic via ``trust_remote_code``) is the wall:

    GPU (MPS) batch-64 : ~11 chunks/s   (best, no swap)
    CPU 1 core         : ~5-13 chunks/s
    CPU 4 workers      : ~8 chunks/s  (and swaps ~8 GB)
    CPU 8 workers      : ~3 chunks/s  (severe swap thrash)

So embedding is **memory-bandwidth bound, not CPU-bound** — multiprocessing makes
it *worse*. The wins that remain:

1. **Batch the GPU** (batch_size 64) instead of per-file small batches.
2. **Overlap** GPU embedding with the single-threaded Chroma insert via a writer
   thread, so the GPU never waits on Chroma.
3. **Larger chunks** (``--max-chars``, default 3600) roughly halve the chunk
   count vs the default 1800 — fine for gemma's 8K context — halving total time.

Reuses the project's chunker, embedder, and Chroma store, so the resulting index
is identical in shape to the standard one and queryable with the same configs.

Usage
-----
    python tools/fast_index.py data/input/10-K data/input/10-Q data/input/8-K
    python tools/fast_index.py data/input/10-K --max-chars 3600 --batch-size 64
"""
from __future__ import annotations

import argparse
import contextlib
import queue
import sys
import threading
import time
from pathlib import Path

from sec_llm_project.sec_intel.chunking.sec_sections import metadata_from_filename
from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.index.builder import IndexBuilder, _infer_filing_type


def iter_chunks(dirs, chunker, default_form):
    for d in dirs:
        base = Path(d)
        for fp in sorted(base.rglob("*.txt")):
            if not fp.is_file():
                continue
            text = fp.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            ft = _infer_filing_type(fp, default_form)
            md = metadata_from_filename(fp, filing_type=ft)
            yield from chunker.chunk_filing(text, md)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dirs", nargs="+")
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--max-chars", type=int, default=3600,
                    help="chunk size (default 3600 = ~2x the standard 1800 -> ~half the chunks)")
    ap.add_argument("--batch-size", type=int, default=64, help="GPU embedding batch size")
    ap.add_argument("--flush", type=int, default=512, help="chunks buffered before an embed batch")
    ap.add_argument("--default-form", default="10-K")
    ap.add_argument("--est-total", type=int, default=0, help="estimated total chunks (for ETA)")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.chunking.max_chars = args.max_chars  # the big lever: fewer, larger chunks

    builder = IndexBuilder(cfg)               # builds the Nomic embedder (mps) + chunker
    index = builder._new_index()              # Chroma-backed SECIndex
    embedder = builder.embedder
    model = embedder._model
    prefix = getattr(embedder, "_d_instr", "")  # "search_document: " for Nomic
    normalize = getattr(embedder, "normalize", True)

    # Resume: skip chunks already persisted (chunk IDs are deterministic), so a
    # restart after interruption continues instead of re-embedding everything.
    existing: set[str] = set()
    if cfg.index.store == "chroma":
        try:
            existing = set(index._chroma_collection.get(include=[])["ids"])
        except Exception:
            existing = set()

    print(f"fast_index: dirs={args.input_dirs} max_chars={args.max_chars} "
          f"batch={args.batch_size} device={model.device} "
          f"resume={len(existing):,} already indexed", flush=True)

    # Writer thread: drains embedded batches into Chroma, overlapping the GPU.
    q: queue.Queue = queue.Queue(maxsize=8)
    inserted = 0

    def writer():
        nonlocal inserted
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            chunks, vecs = item
            index.add(chunks, vecs)
            inserted += len(chunks)
            q.task_done()

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    def encode_resilient(texts, bs):
        # MPS attention memory scales with batch x sequence^2; a batch of unusually
        # long chunks can OOM. Halve the batch and retry rather than crash a long run.
        import torch
        while True:
            try:
                return model.encode(texts, batch_size=bs, normalize_embeddings=normalize,
                                    convert_to_numpy=True)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and bs > 1:
                    with contextlib.suppress(Exception):
                        torch.mps.empty_cache()
                    bs = max(1, bs // 2)
                    print(f"  MPS OOM -> retry batch_size={bs}", flush=True)
                else:
                    raise

    def embed_and_enqueue(buf):
        texts = [prefix + c.text for c in buf]
        vecs = encode_resilient(texts, args.batch_size)
        q.put((buf, [list(map(float, v)) for v in vecs]))

    import torch
    t0 = time.time()
    buf, embedded, batches = [], 0, 0
    last_persist = 0
    skipped = 0
    for chunk in iter_chunks(args.input_dirs, builder.chunker, args.default_form):
        if chunk.chunk_id in existing:
            skipped += 1
            continue
        buf.append(chunk)
        if len(buf) >= args.flush:
            embed_and_enqueue(buf)
            embedded += len(buf)
            buf = []
            batches += 1
            if batches % 8 == 0:
                # Release cached MPS memory so it doesn't build up and force swap,
                # which otherwise degrades the embedding rate over a long run.
                with contextlib.suppress(Exception):
                    torch.mps.empty_cache()
            dt = time.time() - t0
            rate = embedded / dt if dt else 0
            eta = (args.est_total - embedded) / rate / 60 if (rate and args.est_total) else 0
            tail = f" ETA {eta:.0f}min" if eta > 0 else ""
            print(f"[{time.strftime('%H:%M:%S')}] embedded {embedded:,} | inserted {inserted:,} "
                  f"| {rate:.1f} chunks/s{tail}", flush=True)
            if embedded - last_persist >= 20000:
                index.persist()  # periodic sidecar checkpoint
                last_persist = embedded
    if buf:
        embed_and_enqueue(buf)
        embedded += len(buf)

    q.put(None)
    wt.join()
    index.persist()
    dt = time.time() - t0
    print(f"DONE embedding: {embedded:,} chunks in {dt/60:.1f} min "
          f"({embedded/dt:.1f} chunks/s)", flush=True)

    # Finalize: rebuild chunks.jsonl + BM25 from Chroma so the index is
    # query-ready (BM25 precomputed) and correct even if this run auto-restarted.
    with contextlib.suppress(Exception):
        torch.mps.empty_cache()
    from sec_llm_project.sec_intel.index.finalize import finalize_index
    finalize_index(cfg.index.path, collection=cfg.index.collection,
                   embedding_info=embedder.info.to_dict())
    print(f"DONE: index finalized -> {cfg.index.path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
