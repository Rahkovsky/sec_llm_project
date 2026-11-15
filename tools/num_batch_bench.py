"""S3 lab: does a larger Ollama ``num_batch`` speed up prompt processing? (docs/NEXT_STEPS.md S3)

Ollama runs llama.cpp with a 512-token batch. This sends prompts of the size the pipeline
sends (8 filing chunks, ~2.4k tokens, the median of the v6 answer and judge calls) to each
model with ``num_predict`` 1 and reads Ollama's own ``prompt_eval_duration``, so only prompt processing is timed. Every prompt starts
with a fresh nonce, so Ollama's prefix cache cannot reuse an earlier prompt. A changed
``num_batch`` restarts the runner; the first call per setting is a warm-up and is not counted.

    python tools/num_batch_bench.py --models gemma4:12b qwen3:14b --batches 512 1024 2048
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


def prompts(chunks: Path, n: int, per: int, seed: int) -> list[str]:
    texts = []
    with chunks.open() as f:
        for line in f:
            t = json.loads(line)["text"]
            if 800 < len(t) < 1800:  # ~300 tokens: 8 make the ~2.4k-token prompts of v6
                texts.append(t)
            if len(texts) >= 4000:
                break
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        body = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(rng.sample(texts, per)))
        out.append(body + "\n\nQuestion: What do these excerpts say about risk? Answer briefly.")
    return out


def call(url: str, model: str, prompt: str, num_batch: int, num_ctx: int) -> dict:
    payload = {"model": model, "stream": False, "think": False,
               "messages": [{"role": "user", "content": f"Request {random.random():.12f}.\n\n{prompt}"}],
               "options": {"num_predict": 1, "num_ctx": num_ctx, "num_batch": num_batch,
                           "temperature": 0}}
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        b = json.loads(r.read())
    return {"tokens": b.get("prompt_eval_count", 0), "prompt_s": b.get("prompt_eval_duration", 0) / 1e9,
            "load_s": b.get("load_duration", 0) / 1e9, "wall_s": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--models", nargs="+", default=["gemma4:12b", "qwen3:14b"])
    ap.add_argument("--batches", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chunks-per-prompt", type=int, default=8)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--chunks", default="data/sec_index_v4/chunks.jsonl")
    ap.add_argument("--out", default="out/timing/num_batch.json")
    args = ap.parse_args()

    ps = prompts(Path(args.chunks), args.reps, args.chunks_per_prompt, seed=7)
    rows = []
    for model in args.models:
        # interleave settings over two rounds so drift (heat, background load) hits all alike
        for rnd in range(2):
            for nb in args.batches if rnd == 0 else list(reversed(args.batches)):
                call(args.url, model, ps[0], nb, args.num_ctx)  # warm-up / runner restart
                for i, p in enumerate(ps):
                    r = call(args.url, model, p, nb, args.num_ctx)
                    rows.append({"model": model, "num_batch": nb, "round": rnd, "prompt": i, **r})
                    print(f"{model} b={nb} r{rnd} p{i}: {r['tokens']} tok in {r['prompt_s']:.1f}s "
                          f"({r['tokens'] / max(r['prompt_s'], 1e-9):.0f} tok/s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))

    print("\n| model | num_batch | prompt tokens (median) | prompt s (median) | tok/s (median) |")
    print("|---|---|---|---|---|")
    for model in args.models:
        for nb in args.batches:
            rs = [r for r in rows if r["model"] == model and r["num_batch"] == nb]
            print(f"| {model} | {nb} | {statistics.median(r['tokens'] for r in rs):.0f} | "
                  f"{statistics.median(r['prompt_s'] for r in rs):.2f} | "
                  f"{statistics.median(r['tokens'] / r['prompt_s'] for r in rs):.0f} |")


if __name__ == "__main__":
    main()
