#!/usr/bin/env python3
"""Parallel-range downloader for large Hugging Face GGUF files.

Why this exists
---------------
``ollama pull`` and the default ``huggingface_hub`` download both use a *single*
HTTP stream. On a high-latency link (e.g. a cellular hotspot with ~1 s RTT) a
single stream is capped by the TCP bandwidth-delay product to a tiny fraction of
the real bandwidth — we measured **40 KB/s single-stream vs 4.5 MB/s with 48
parallel streams** on the same connection, a 100x difference.

Hugging Face's Xet "high performance" mode does not help here: it requires
**64 GB RAM** for its buffers (it degrades on less), and its chunked
reconstruction protocol stalled entirely on our test connection.

This tool does what actually works: many concurrent HTTP range requests against
the resolved CDN URL, written directly into one preallocated file (each worker
seeks to its own offset — no reassembly step). It resumes across restarts and
re-signs the CDN URL when it expires.

Usage
-----
    # Authenticate first (lifts the anonymous per-IP rate limit that throttles
    # many parallel streams). A free *read* token is enough:
    #   https://huggingface.co/settings/tokens
    export HF_TOKEN=hf_...

    python tools/download_gguf.py \
        --repo MaziyarPanahi/gemma-3-27b-it-GGUF \
        --file gemma-3-27b-it.Q4_K_M.gguf \
        --out ~/.ollama/imports/gemma-3-27b-it.Q4_K_M.gguf \
        --concurrency 48

Tune ``--concurrency`` up until throughput stops improving (it plateaus at the
connection's real ceiling). 16-48 is a good range.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_url_lock = threading.Lock()
_done_lock = threading.Lock()
_signed: dict[str, object] = {"url": None, "ts": 0.0}


def resolve_url(resolve_endpoint: str, token: str) -> str:
    """HEAD the resolve endpoint and capture the 302 signed CDN URL.

    Note: do NOT add a Range header to the resolve request — it corrupts the
    signature and the CDN then returns 403.
    """
    captured: dict[str, str | None] = {"url": None}

    class Capture(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
            captured["url"] = newurl

    opener = urllib.request.build_opener(Capture)
    req = urllib.request.Request(resolve_endpoint, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        opener.open(req, timeout=30)
    except urllib.error.HTTPError as exc:
        if not captured["url"]:
            captured["url"] = exc.headers.get("Location")
    if not captured["url"]:
        raise RuntimeError("could not resolve CDN URL from resolve endpoint")
    return captured["url"]  # type: ignore[return-value]


def total_size(resolve_endpoint: str, token: str) -> int:
    req = urllib.request.Request(resolve_endpoint, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        size = r.headers.get("x-linked-size") or r.headers.get("Content-Length")
    if not size:
        raise RuntimeError("server did not report file size")
    return int(size)


def current_url(resolve_endpoint: str, token: str) -> str:
    with _url_lock:
        if _signed["url"] is None or time.time() - float(_signed["ts"]) > 1800:
            _signed["url"] = resolve_url(resolve_endpoint, token)
            _signed["ts"] = time.time()
        return _signed["url"]  # type: ignore[return-value]


def refresh_url(resolve_endpoint: str, token: str) -> None:
    with _url_lock:
        _signed["url"] = resolve_url(resolve_endpoint, token)
        _signed["ts"] = time.time()


def mark_done(progress_path: str, idx: int) -> None:
    with _done_lock, open(progress_path, "a") as f:
        f.write(f"{idx}\n")


def load_done(progress_path: str) -> set[int]:
    if os.path.exists(progress_path):
        return {int(x) for x in open(progress_path).read().split() if x.strip()}
    return set()


def download_chunk(resolve_endpoint: str, token: str, dest: str, progress_path: str,
                   idx: int, start: int, end: int) -> int:
    length = end - start + 1
    for attempt in range(8):
        try:
            req = urllib.request.Request(current_url(resolve_endpoint, token))
            req.add_header("Range", f"bytes={start}-{end}")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            got = 0
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "r+b") as f:
                f.seek(start)
                while True:
                    buf = r.read(256 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
            if got == length:
                mark_done(progress_path, idx)
                return length
            raise OSError(f"short read {got}/{length}")
        except Exception as exc:  # retrying on any failure is the whole point
            if attempt >= 3:
                print(f"  chunk {idx} attempt {attempt}: {type(exc).__name__}: {str(exc)[:100]}",
                      flush=True)
            if attempt >= 2:
                refresh_url(resolve_endpoint, token)  # signed URL may have expired
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"chunk {idx} failed after retries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. MaziyarPanahi/gemma-3-27b-it-GGUF")
    ap.add_argument("--file", required=True, help="filename within the repo (the .gguf)")
    ap.add_argument("--out", required=True, help="destination path on disk")
    ap.add_argument("--concurrency", type=int, default=16, help="parallel streams (default 16)")
    ap.add_argument("--chunk-mb", type=int, default=16, help="chunk size in MB (default 16)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                    help="HF token (default: $HF_TOKEN). A read token lifts the anon rate limit.")
    args = ap.parse_args()

    resolve_endpoint = f"https://huggingface.co/{args.repo}/resolve/main/{args.file}"
    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    progress_path = dest + ".progress"
    token = args.token.strip()
    chunk = args.chunk_mb * 1024 * 1024

    total = total_size(resolve_endpoint, token)
    print(f"{args.file}: {total/1e9:.2f} GB | {args.concurrency} streams x {args.chunk_mb}MB "
          f"| auth={'yes' if token else 'no'}", flush=True)

    # Preallocate once (sparse). `du` then tracks real downloaded bytes.
    if not os.path.exists(dest) or os.path.getsize(dest) != total:
        with open(dest, "wb") as f:
            f.truncate(total)
        if os.path.exists(progress_path):
            os.remove(progress_path)

    chunks, idx, pos = [], 0, 0
    while pos < total:
        end = min(pos + chunk - 1, total - 1)
        chunks.append((idx, pos, end))
        pos, idx = end + 1, idx + 1

    done = load_done(progress_path)
    todo = [c for c in chunks if c[0] not in done]
    print(f"chunks: {len(chunks)} total, {len(done)} done, {len(todo)} to fetch", flush=True)

    t0 = time.time()
    base = len(done) * chunk
    done_bytes = base
    n = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(download_chunk, resolve_endpoint, token, dest, progress_path, i, s, e)
                for (i, s, e) in todo]
        for fut in as_completed(futs):
            done_bytes += fut.result()
            n += 1
            if n % 5 == 0 or n == len(todo):
                dt = time.time() - t0
                rate = (done_bytes - base) / dt if dt > 0 else 0
                eta = (total - done_bytes) / rate / 60 if rate > 0 else 0
                print(f"[{time.strftime('%H:%M:%S')}] {done_bytes/1e9:.2f}/{total/1e9:.2f} GB "
                      f"({100*done_bytes/total:.0f}%) {rate/1e6:.2f} MB/s ETA {eta:.0f}min", flush=True)

    final = os.path.getsize(dest)
    if final != total:
        print(f"ERROR: final size {final} != expected {total}", flush=True)
        return 1
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"DONE: {final/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
