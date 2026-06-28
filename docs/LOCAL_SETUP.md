# Local Stack — Setup & Status

A living runbook for running the SEC Disclosure Intelligence Prototype **fully
locally** (no API keys, no data leaving the host): Ollama **gemma3:12b** for
generation + **Nomic** embeddings. It documents every step end-to-end *and*
tracks the current state of this deployment.

> All commands below are **paste-safe**: one command per line, no inline `#`
> comments (those break in zsh without `interactive_comments`).

---

## Current status

_Last updated: 2026-06-26._

| Component | State | Detail |
| --- | --- | --- |
| Ollama server | ✅ running | v0.30.10 at `http://localhost:11434` (started by the macOS app) |
| Generator (LLM) | ✅ `gemma3:12b` | 7.3 GB Q4, ~12–18 tok/s, runs 100% on GPU, no swap |
| Embeddings | ✅ Nomic (pinned) | `nomic-embed-text-v1.5` @ `e9b6763`, dim 768 |
| Filings (data) | ✅ 4,199 files | S&P 500 10-K/10-Q/8-K, 2023–2025, 848 MB under `data/input/` |
| Index | 🔄 building (overnight) | **Full corpus** via `tools/fast_index.py`, 3600-char chunks (~292k); started 2026-06-26; ~10–11 h; auto-restart + resume |
| Index profile | `config/local.yaml` | Chroma store + Nomic embeddings, `data/sec_index/chroma_db` |

When the index finishes, flip "Index" to ✅ and record the chunk count. The index
is **additive** — add more companies/forms later without redoing existing work.

---

## Step 1 — Install the model (`gemma3:12b`)

**1a. Install Ollama** (native binary, not a Python package):
```
brew install ollama
```
On macOS the app starts the server automatically. Only run `ollama serve` if the
server is actually down (`curl http://localhost:11434/api/version` to check).

**1b. Get the model.** The simple path:
```
ollama pull gemma3:12b
```

**If `ollama pull` is slow or stalls** (high-latency / hotspot connections),
download the GGUF with the parallel-range tool and register it manually:
```
export HF_TOKEN=hf_your_read_token
python tools/download_gguf.py --repo MaziyarPanahi/gemma-3-12b-it-GGUF --file gemma-3-12b-it.Q4_K_M.gguf --out ~/.ollama/imports/gemma-3-12b-it.Q4_K_M.gguf --concurrency 48
printf 'FROM %s/.ollama/imports/gemma-3-12b-it.Q4_K_M.gguf\nPARAMETER num_ctx 8192\n' "$HOME" > Modelfile
ollama create gemma3:12b -f Modelfile
```
A free HF **read** token (https://huggingface.co/settings/tokens) lifts the
anonymous rate limit when running many parallel streams. Do **not** set
`HF_XET_HIGH_PERFORMANCE=1` — it needs 64 GB RAM.

> **RAM note:** `gemma3:12b` (~7 GB) fits 16–32 GB comfortably. `gemma3:27b`
> (~17 GB) needs ~48 GB+ — on 32 GB it swaps to disk and crawls at ~0.1 tok/s.

---

## Step 2 — Python deps + embeddings

```
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[embeddings,vectorstore,yaml]"
```

This installs the Nomic embedding stack (sentence-transformers + einops) and
Chroma. The default Nomic model is **revision-pinned** in `config/local.yaml`
(`trust_remote_code` runs a reviewed, fixed commit). No further setup needed.

---

## Step 3 — Get the filings (already done here)

The corpus under `data/input/` was downloaded from official SEC EDGAR
(rate-limited, compliant User-Agent). To reproduce or extend it:
```
export SEC_USER_NAME="Your Name"
export SEC_USER_EMAIL="you@org.gov"
uv pip install -e ".[sec]"
sec-intel fetch --sp500 --forms 10-K 10-Q 8-K --years 2023-2025
```
Filings land in `data/input/<FORM>/<TICKER>/`. Current state: **4,199 files**
(1,396 × 10-K, 1,399 × 10-Q, 1,401 × 8-K).

---

## Step 4 — Build the full index

The index is **additive and idempotent** (chunks are `upsert`-ed to Chroma as
they are embedded — interruptions don't lose work, re-runs don't duplicate), so
the practical pattern is to **index the companies you care about first, then
expand**.

**Recommended: major companies first** (what this deployment did — ~1–2 h):
```
sec-intel --config config/local.yaml index data/input/10-K/AAPL data/input/10-K/MSFT data/input/10-K/NVDA
```
Pass as many `data/input/10-K/<TICKER>` folders as you want on one line.

**Expand later** (additive — does not redo existing companies):
```
sec-intel --config config/local.yaml index data/input/10-K/JPM data/input/10-Q/AAPL
```

### Full corpus — `tools/fast_index.py`

For the **whole corpus** (all 4,199 filings, ~292k chunks at 3600-char), use the
optimized indexer. Profiling on an M1 Pro / 32 GB found the **embedding model is
the wall**, not Chroma or chunking:

| Stage | Rate | Note |
| --- | --- | --- |
| Chunking | ~20,000 chunks/s | free |
| Chroma insert | ~900 vec/s @ 16k | not the bottleneck |
| **Embedding (Nomic, MPS)** | **~8–11 chunks/s** | **the wall** |

Counter-intuitively, **multiprocessing makes embedding *slower*** — it's
memory-bandwidth bound, so 8 CPU workers (3/s, swapping) lose to a single GPU
stream (11/s). `tools/fast_index.py` therefore uses one GPU stream with: batched
embedding, an overlapped Chroma writer thread, periodic MPS cache clears (stops a
memory-creep slowdown), OOM-resilient batches, 2× chunk size (halves the count),
and **resume** (skips already-embedded chunks on restart).

```
python tools/fast_index.py data/input/10-K data/input/10-Q data/input/8-K --max-chars 3600 --batch-size 16
```

Expect **~10–11 h** for the full corpus. To run it **overnight with the lid
closed** (macOS): keep it on the charger, prevent sleep, and wrap it so it
auto-resumes if it ever crashes:

```
sudo pmset -c disablesleep 1
nohup caffeinate -dimsu bash -c 'until python tools/fast_index.py data/input/10-K data/input/10-Q data/input/8-K --max-chars 3600 --batch-size 16; do sleep 20; done' > /tmp/overnight.log 2>&1 &
```
Check progress with `grep chunks/s /tmp/overnight.log | tail`. When done, restore
sleep: `sudo pmset -c disablesleep 0`.

Output persists to `data/sec_index/chroma_db` (path set in `config/local.yaml`).

**Finalization (automatic).** When embedding finishes, `fast_index.py` runs a
finalize step that rebuilds `chunks.jsonl` and a persisted **`bm25.json`** lexical
index *from Chroma* (the complete, restart-proof source). This (a) makes the index
correct even if the run auto-restarted, and (b) precomputes BM25 so it isn't
re-tokenized on the first query of every session (warmup drops from ~36s to ~6s).
If a run was interrupted before finalizing, run it standalone:
```
python tools/finalize_index.py --config config/local.yaml
```

---

## Step 5 — Ask questions (use it)

Once indexed, query entirely locally. No re-indexing between questions.
```
sec-intel --config config/local.yaml ask "What are Apple's most significant risk factors?" --ticker AAPL
```
```
sec-intel --config config/local.yaml ask "What legal proceedings does Microsoft disclose?" --ticker MSFT
```

Flagship year-over-year disclosure monitor, structured extraction, and diff:
```
sec-intel --config config/local.yaml monitor AAPL 2024 2025
```
```
sec-intel --config config/local.yaml extract risk_factors --ticker MSFT
```
```
sec-intel --config config/local.yaml compare AAPL 2024 2025 --items 1A 3 7
```

Flags: `--ticker` scopes to one company, `--top-k N` widens the evidence pool,
`--json` emits machine-readable output (pipe to `jq`).

**Latency:** first question after idle adds a one-time ~5 s model load; after
that, answers land in ~3–10 s (longer for risk-factor questions that pull many
large chunks into context).

---

## Notes / gotchas

- **Don't double-run `ollama serve`** — the macOS app already runs the server;
  a second one errors with "address already in use".
- **zsh + inline comments:** never paste a command with a trailing `# comment` —
  zsh without `interactive_comments` passes it as arguments (and backticks in a
  comment get executed). All snippets here avoid that.
- **Keep the model resident:** set `keep_alive: -1` (or query with that option)
  to avoid the ~5 s reload between idle periods.
- **Switching models:** change `llm.model` in `config/local.yaml`, or override
  with `export SECI_LLM__MODEL=...`.

See also: [COMMANDS.md](COMMANDS.md) (full CLI reference) and
[FUTURE_WORK.md](FUTURE_WORK.md).
