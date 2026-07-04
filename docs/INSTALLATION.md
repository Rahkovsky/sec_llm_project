# 1. Installation — packages, data, and indexing

Everything you need **before** running the system for real: install the Python
packages and the LLM stack, download the filings, and build the index. When
you're done here, go to **[3. Real Run](REAL_RUN.md)**.

> Only kicking the tyres with the deterministic mock (no model, no data)? You can
> skip straight to **[2. Mock Run](MOCK_RUN.md)** — it needs only Step 1a below.

Contents:
1. [Python environment + core](#1a-python-environment--core)
2. [LLM stack (required for real answers)](#1b-llm-stack-required-for-real-answers)
3. [Download the filings](#1c-download-the-filings)
4. [Build the index](#1d-build-the-index)

---

## 1a. Python environment + core

```bash
# Install uv (fast Python package manager — https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
source $HOME/.local/bin/env                        # add uv to PATH (this shell)
# brew install uv                                  # macOS Homebrew alternative
# pip install uv                                   # any platform

# Python 3.13 + virtualenv
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate

# Core platform (stdlib only — enough for tests and `sec-intel-mock`)
uv pip install -e .
```

## 1b. LLM stack (required for real answers)

The real stack is **local**: an Ollama-served model for generation + U.S.-origin
Nomic embeddings. No API keys, no data leaves the host.

```bash
# 1. Ollama — native binary, NOT a Python package
brew install ollama
brew services start ollama                                

# 2. Pull the generation model (~7 GB Q4; fits 16–32 GB RAM comfortably)
ollama pull gemma3:12b
 # optional independent  judge (citation verification)
ollama pull mistral:7b                         

# 3. Python backends the real stack needs: Nomic embeddings + Chroma vector store
uv pip install -e ".[embeddings,vectorstore,yaml]"

# 4. EDGAR ingestion (needed only for the download step below)
uv pip install -e ".[sec]"
```

Confirm the model is reachable before indexing:

```bash
ollama list          # should list gemma3:12b
```

> **RAM note:** `gemma3:12b` (~7 GB) fits 16–32 GB. `gemma3:27b` (~17 GB) needs
> ~48 GB+ — on 32 GB it swaps to disk and crawls (~0.1 tok/s).

## 1c. Download the filings

Filings come from official SEC EDGAR (rate-limited, compliant User-Agent). Set
your identity, then fetch:

```bash
export SEC_USER_NAME="Your Name"
export SEC_USER_EMAIL="you@org.gov"

# A few companies, multi-form, multi-year (2+ consecutive years enable the monitor):
sec-intel fetch MSFT TSLA --forms 10-K 10-Q 8-K --years 2023-2025

# ...or the full S&P 500 universe:
sec-intel fetch --sp500 --forms 10-K 10-Q 8-K --years 2023-2025
```

Filings land under `data/input/<FORM>/<TICKER>/`.

## 1d. Build the index

Indexing embeds every chunk (Nomic) and writes them to Chroma. It is **additive
and idempotent** — chunks are `upsert`-ed as they're embedded, so interruptions
don't lose work and re-runs don't duplicate. Practical pattern: index the
companies you care about first, then expand.

> **Index and query with the same config.** The embedder that *built* the index
> and the one used to *query* must match, or you'll get an `Embedding mismatch`
> error. Use `config/local.yaml` for both indexing and querying.

```bash
# Targeted (fast): a few companies
sec-intel --config config/local.yaml index data/input/10-K/AAPL data/input/10-K/MSFT

# Expand later (additive — does not redo existing companies)
sec-intel --config config/local.yaml index data/input/10-K/JPM data/input/10-Q/AAPL
```

### Full corpus — `tools/fast_index.py`

For the whole corpus (all filings, ~292k chunks at 3600-char), use the optimized
indexer. Profiling (M1 Pro / 32 GB) shows the **embedding model is the wall**, not
Chroma or chunking, and multiprocessing makes it *slower* (memory-bandwidth
bound). So `fast_index.py` uses a single GPU stream with batching, an overlapped
Chroma writer, periodic MPS cache clears, OOM-resilient batches, 2× chunk size,
and **resume**.

```bash
python tools/fast_index.py data/input/10-K data/input/10-Q data/input/8-K --max-chars 3600 --batch-size 16
```

Expect **~10–11 h** for the full corpus. Overnight (macOS), auto-resuming on crash:

```bash
sudo pmset -c disablesleep 1
nohup caffeinate -dimsu bash -c 'until python tools/fast_index.py data/input/10-K data/input/10-Q data/input/8-K --max-chars 3600 --batch-size 16; do sleep 20; done' > /tmp/overnight.log 2>&1 &
# progress: grep chunks/s /tmp/overnight.log | tail
# when done: sudo pmset -c disablesleep 0
```

Output persists to `data/sec_index/chroma_db` (path set in `config/local.yaml`).

**Finalization (automatic).** When embedding finishes, `fast_index.py` rebuilds
`chunks.jsonl` and a persisted **`bm25.json`** lexical index *from Chroma* — making
the index correct even after an auto-restart and precomputing BM25 so it isn't
re-tokenized on the first query (warmup drops from ~36 s to ~6 s). If a run was
interrupted before finalizing:

```bash
python tools/finalize_index.py --config config/local.yaml
```

---

**Next:** [3. Real Run](REAL_RUN.md) · full CLI reference in [COMMANDS.md](COMMANDS.md).
