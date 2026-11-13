"""Repair metadata of an index built by ``tools/fast_index.py`` (no re-embedding).

``fast_index.py`` derived chunk metadata from the filename only, ignoring the
per-filing ``.json`` sidecar. Result: company=UNKNOWN, filing_date=<year only>,
fiscal_year=<filing year> and an empty source_url on every chunk it embedded.
Chunk text / item_number / section_title are identical either way (verified), so
this script rewrites only the metadata from the sidecar, in place, and (optionally)
removes duplicate chunks left by an earlier ``sec-intel index`` run whose IDs differ.

Dry-run by default. Back up the index directory first, then pass --apply:

    cp -R data/sec_index_qwen06 data/sec_index_qwen06.bak
    python tools/repair_metadata.py --config config/local_qwen06.yaml            # dry run
    python tools/repair_metadata.py --config config/local_qwen06.yaml --apply
    python tools/finalize_index.py  --config config/local_qwen06.yaml            # rebuild bm25/chunks.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sec_llm_project.sec_intel.chunking.sec_sections import SECChunker, metadata_from_filename
from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.index.builder import _infer_filing_type, _load_sidecar

FIELDS = ("company", "cik", "filing_date", "fiscal_year", "source_url", "filing_type")


def expected_chunks(dirs, chunker):
    """Replicates fast_index's chunk generation: id -> (source_path, text)."""
    ids, texts = set(), defaultdict(set)
    for d in dirs:
        for fp in sorted(Path(d).rglob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            md = metadata_from_filename(fp, filing_type=_infer_filing_type(fp, "10-K"))
            for c in chunker.chunk_filing(text, md):
                ids.add(c.chunk_id)
                texts[str(fp)].add(c.text)
    return ids, texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dirs", nargs="*", default=["data/input/10-K", "data/input/10-Q", "data/input/8-K"])
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-delete-duplicates", action="store_true")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.chunking.max_chars = args.max_chars
    import chromadb
    client = chromadb.PersistentClient(path=str(Path(cfg.index.path) / "chroma_db"))
    col = client.get_collection(cfg.index.collection)
    n = col.count()
    print(f"collection {cfg.index.collection}: {n:,} chunks  mode={'APPLY' if args.apply else 'DRY-RUN'}")

    print("computing expected chunk ids (same chunker as fast_index)...", flush=True)
    exp_ids, exp_texts = expected_chunks(args.input_dirs, SECChunker(cfg.chunking))
    print(f"  expected {len(exp_ids):,} chunk ids")

    sidecars: dict[str, dict | None] = {}

    def sidecar_md(src: str):
        if src not in sidecars:
            sc = _load_sidecar(Path(src))
            sidecars[src] = sc["metadata"] if sc else None
        return sidecars[src]

    to_update_ids, to_update_md, orphans, stats = [], [], [], Counter()
    offset = 0
    while offset < n:
        got = col.get(limit=args.batch, offset=offset, include=["metadatas", "documents"])
        for cid, md, doc in zip(got["ids"], got["metadatas"], got["documents"]):
            stats["seen"] += 1
            src = md.get("source_path", "")
            if cid not in exp_ids:
                # chunk from a different ID scheme (earlier sec-intel index run)
                if doc in exp_texts.get(src, ()):
                    orphans.append(cid); stats["duplicate_orphans"] += 1
                else:
                    stats["orphan_text_not_covered"] += 1   # never deleted
                continue
            if md.get("company") not in (None, "", "UNKNOWN"):
                stats["already_rich"] += 1
                continue
            side = sidecar_md(src)
            if not side:
                stats["no_sidecar"] += 1
                continue
            new = dict(md)
            for k in FIELDS:
                if side.get(k) not in (None, ""):
                    new[k] = side[k]
            if new != md:
                to_update_ids.append(cid); to_update_md.append(new); stats["to_update"] += 1
        offset += args.batch
        if (offset // args.batch) % 50 == 0:
            print(f"  scanned {min(offset, n):,}/{n:,}", flush=True)

    print("summary:", dict(stats))
    if not args.apply:
        print("DRY-RUN: nothing changed. Re-run with --apply.")
        return 0
    for i in range(0, len(to_update_ids), args.batch):
        col.update(ids=to_update_ids[i:i + args.batch], metadatas=to_update_md[i:i + args.batch])
    print(f"updated metadata on {len(to_update_ids):,} chunks")
    if orphans and not args.no_delete_duplicates:
        for i in range(0, len(orphans), args.batch):
            col.delete(ids=orphans[i:i + args.batch])
        print(f"deleted {len(orphans):,} duplicate chunks")
    print(f"collection now {col.count():,} chunks. Next: python tools/finalize_index.py --config {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
