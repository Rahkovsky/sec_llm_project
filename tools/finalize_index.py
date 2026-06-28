#!/usr/bin/env python3
"""Finalize a built index: rebuild chunks.jsonl + BM25 from Chroma.

Run once after indexing completes (fast_index.py does this automatically, but
this lets you re-run it standalone — e.g. if an indexing run was interrupted and
auto-restarted, so its in-memory chunk set was partial).

    python tools/finalize_index.py --config config/local.yaml
"""
from __future__ import annotations

import argparse
import sys

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.index.finalize import finalize_index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/local.yaml")
    ap.add_argument("--path", default=None, help="index path (default: from config)")
    args = ap.parse_args()
    cfg = AppConfig.load(args.config)
    finalize_index(args.path or cfg.index.path, collection=cfg.index.collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
