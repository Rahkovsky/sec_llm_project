# Commands: test & run reference

Complete, copy-pasteable commands for the **SEC Disclosure Intelligence
Prototype**. Everything under "Offline" needs no API keys, no model downloads,
and no network — it uses the deterministic mock LLM + hashing embeddings.

- [1. Setup](#1-setup)
- [2. Test, lint, type-check](#2-test-lint-type-check)
- [3. Run offline (no keys)](#3-run-offline-no-keys)
- [4. Run with real filings + a real model backend](#4-run-with-real-filings--a-real-model-backend)
- [5. Ingest official SEC filings (download + process locally)](#5-ingest-official-sec-filings-download--process-locally)
- [6. Docker](#6-docker)
- [7. Python API](#7-python-api)
- [8. Config & environment](#8-config--environment)
- [9. Verified citations & faithfulness harness](#9-verified-citations--faithfulness-harness)

---

## 1. Setup

### 1a. uv + Python

```bash
# Install uv — cross-platform Python package manager (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
source $HOME/.local/bin/env                           # add uv to PATH (current shell)
# Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex

# Install Python 3.13 (uv manages Python versions — no system Python needed)
uv python install 3.13

# Create a virtual environment and activate it
uv venv --python 3.13
source .venv/bin/activate                             # macOS / Linux
# Windows: .venv\Scripts\activate

# Core platform only (standard library — enough for tests, CI, and the offline demo)
uv pip install -e .

# Optional backends (install only what you need)
uv pip install -e ".[embeddings]"    # local Nomic embeddings via sentence-transformers
uv pip install -e ".[vectorstore]"   # Chroma persistent vector store
uv pip install -e ".[sec]"           # EDGAR ingestion (edgartools)
uv pip install -e ".[observability]" # span tracing -> Arize Phoenix (optional)
uv pip install -e ".[dev,yaml]"      # pytest/ruff/pyright + YAML config support
uv pip install -e ".[all,dev]"       # everything
```

### 1b. Local / free model stack (no API keys)

The `config/local.yaml` profile runs entirely on your machine using
**Nomic embeddings** (sentence-transformers) and **Ollama** for generation.
No data leaves the host.

> For the staged local runbook, see **[INSTALLATION.md](INSTALLATION.md)** (model
> install, deps, data, full-corpus indexing) and **[REAL_RUN.md](REAL_RUN.md)**
> (querying).

**Step 1 — Python packages** (uv handles these):

```bash
uv pip install -e ".[embeddings,vectorstore,yaml]"
```

**Step 2 — Ollama** (a native binary — not a Python package, installed separately):

```bash
# macOS (recommended)
brew install ollama
# or download the .dmg from https://ollama.com/download/mac

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download the installer from https://ollama.com/download/windows
```

**Step 3 — Get the model and start the server:**

```bash
ollama pull gemma3:12b      # download once (~7 GB) — fits 16-32 GB RAM
ollama serve                # keep running in a separate terminal
```

> **Model sizing (RAM matters):** `gemma3:12b` (~7 GB Q4) runs comfortably on
> 16-32 GB. `gemma3:27b` (~17 GB) is higher quality but needs **~48 GB+** — on a
> 32 GB machine it swaps to disk and slows to ~0.1 tok/s (minutes per answer).
> Pick 12b unless you have the RAM headroom.

**Step 4 — Index and query:**

```bash
# Index all downloaded forms (uses Nomic embeddings locally)
sec-intel --config config/local.yaml index data/input/10-K data/input/10-Q data/input/8-K

# Ask a grounded question (uses Ollama gemma3:12b locally)
sec-intel --config config/local.yaml ask "What are Tesla's main risk factors?" --ticker TSLA
```

> **Speed note:** Nomic embeddings run on CPU or GPU automatically.
> On Apple Silicon (M-series) and NVIDIA GPUs, sentence-transformers uses
> hardware acceleration — expect 1-3 h for ~4,200 filings.
> On CPU-only machines the same run takes significantly longer; consider
> indexing one form at a time (`index data/input/10-K`) to validate first.

## 2. Test, lint, type-check

```bash
pytest -q                          # full offline suite (deterministic; ~0.1s)
pytest -q tests/test_monitor.py    # a single test file
pytest -q -k retrieval             # tests matching a keyword
pytest -q -v                       # verbose, per-test names

ruff check src tests               # lint
ruff format src tests              # auto-format
pyright src                        # type-check (strict; configured in pyrightconfig.json)
```

Run exactly what CI runs (`.github/workflows/ci.yml`):

```bash
uv pip install -e ".[dev,yaml]"
ruff check src tests
pytest -q
sec-intel-mock demo
sec-intel-mock evaluate --demo
```

## 3. Run offline (no keys)

These use the built-in synthetic corpus and the offline mock backend.

```bash
# End-to-end demo: grounded answer with citations (the demo scopes to NOVA)
sec-intel-mock demo
sec-intel-mock demo --question "What cybersecurity risks does Nova Devices face?"

# Flagship: Disclosure Change & Risk Signal Monitor
sec-intel-mock monitor NOVA 2023 2024 --demo
sec-intel-mock monitor NOVA 2023 2024 --demo | jq '.summary'

# Evaluation: retrieval + faithfulness metrics
sec-intel-mock evaluate --demo
sec-intel-mock evaluate --demo --json

# Snapshot: diff current metrics against the committed baseline
sec-intel-mock evaluate --demo --snapshot                        # print per-metric deltas
sec-intel-mock evaluate --demo --update-baseline                 # rewrite baseline.json

# Enforcement-case benchmark (evaluation, not training)
sec-intel-mock evaluate --demo --enforcement
sec-intel-mock evaluate --demo --enforcement --json
```

You can also force the offline profile explicitly with any subcommand:

```bash
sec-intel --config config/offline.json <subcommand> ...
```

## 4. Run with real filings (local stack)

First **activate the venv** so `sec-intel` is on your PATH — otherwise you get
`zsh: command not found: sec-intel`:

```bash
source .venv/bin/activate
```

Everything runs locally with `config/local.yaml` (Ollama `gemma3:12b` + Nomic
embeddings — **start Ollama first**, see section 1b). No API keys.

> **Match the embedder to the index.** `ask`/`monitor`/`extract`/`compare` must use
> the **same config you built the index with**. Querying a Nomic-built index with a
> different embedder fails with `Embedding mismatch: index was built with '...' but
> query uses '...'`. Use `config/local.yaml` everywhere below.

```bash
# 1) Build an index (one or more form directories — all combined into one index)
sec-intel --config config/local.yaml index data/input/10-K data/input/10-Q data/input/8-K
sec-intel --config config/local.yaml index data/input/10-K
sec-intel --config config/local.yaml index data/input/10-K --pattern "*.txt" --max-files 50

# 2) Ask a grounded question (cited, with confidence + abstention)
sec-intel --config config/local.yaml ask \
  "Summarize liquidity risks for Tesla with citations." --ticker TSLA
sec-intel --config config/local.yaml ask "How did risk factors change?" --ticker MSFT --top-k 10 --json

# 3) Compare two filing years
sec-intel --config config/local.yaml compare MSFT 2024 2025
sec-intel --config config/local.yaml compare MSFT 2024 2025 --items 1A 3 7

# 4) Run the disclosure monitor across forms
sec-intel --config config/local.yaml monitor MSFT 2024 2025 --forms 10-K 10-Q 8-K
sec-intel --config config/local.yaml monitor MSFT 2024 2025 --xbrl   # +XBRL context (needs network)

# 5) Structured, citation-grounded extraction
sec-intel --config config/local.yaml extract risk_factors --ticker MSFT
# targets: risk_factors | litigation | mdna | internal_controls | related_party | liquidity

# 6) Evaluate over your own corpus + cases
sec-intel --config config/local.yaml evaluate --input-dir data/input/10-K --cases my_cases.json
```

For a fully offline run (no Ollama, mock model — tests/demos):

```bash
sec-intel --config config/offline.yaml demo
```

## 5. Ingest official SEC filings (download + process locally)

Rate-limited (≤ 10 req/s, SEC fair-access) with a compliant User-Agent built
from `SEC_USER_NAME` / `SEC_USER_EMAIL`. Filings are saved as plain text under
`data/input/<FORM>/<TICKER>/` so the indexer can recover the form type and year.

**Data strategy:** the comparison engine and the Disclosure Monitor diff filings
*year over year*, so download **at least two consecutive years** and the forms
you care about (`10-K` annual, `10-Q` quarterly, `8-K` events, `DEF 14A` proxy).

```bash
export SEC_USER_NAME="Your Name"
export SEC_USER_EMAIL="you@org.gov"
uv pip install -e ".[sec]"

# (a) Download — multiple tickers, forms, and years in one rate-limited pass.
sec-intel fetch MSFT TSLA --forms 10-K 10-Q 8-K --years 2023-2025
sec-intel fetch AAPL --forms 10-K --years 2021-2025          # a range works too
sec-intel fetch NVDA --forms 10-K DEF\ 14A                   # latest only (no --years)

# Full S&P 500 (bundled Q1-2025 list, approx 500 companies):
# 10-K only: approx 1,500 filings, 0.4 GB, 1 h
sec-intel fetch --sp500 --forms 10-K --years 2023-2025
# All three forms: approx 4,500 filings, 1 GB, 3 h
sec-intel fetch --sp500 --forms 10-K 10-Q 8-K --years 2023-2025

# Live S&P 500 (fetches current Wikipedia list; requires network):
sec-intel fetch --sp500-live --forms 10-K --years 2024-2025

# (b) Process — build a unified index from one or more form directories.
# Single form:
sec-intel --config config/local.yaml index data/input/10-K
# All three forms in one index (recommended for monitor/compare):
sec-intel --config config/local.yaml index data/input/10-K data/input/10-Q data/input/8-K

# (c) Use — the year-over-year flagship now has the data it needs.
sec-intel --config config/local.yaml monitor MSFT 2024 2025 --forms 10-K 10-Q 8-K
sec-intel --config config/local.yaml compare MSFT 2024 2025 --items 1A 3 7
```

Equivalent Python API (same rate limiting and output layout):

```bash
python -c "from sec_llm_project.sec_intel.ingest import download_filings; \
print(download_filings(['MSFT','TSLA'], ['10-K','10-Q','8-K'], years=[2023, 2024, 2025]))"
```

> **Note:** the EDGAR extractor fetches one filing per form per year, which is
> exactly right for the annual `10-K` year-over-year comparison. For `10-Q`/`8-K`
> (multiple per year) it takes the first match in each year — enough to enrich
> `ask`, and you can widen `--years` to deepen the corpus.

## 6. Docker

```bash
# Build the minimal offline image and run the demo
docker build -t sec-intel .
docker run --rm sec-intel                         # runs `sec-intel-mock demo`
docker run --rm sec-intel-mock monitor NOVA 2023 2024 --demo

# Build with the local backends (Nomic embeddings + Chroma). Ollama runs on the
# host; point the container at it with OLLAMA_BASE_URL.
docker build --build-arg EXTRAS="embeddings,vectorstore,yaml" -t sec-intel:local .
docker run --rm -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v "$PWD/data:/app/data" sec-intel:local \
  --config config/local.yaml ask "..." --ticker MSFT
```

## 7. Python API

```python
from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.pipeline import SECIntelPipeline

pipeline = SECIntelPipeline(AppConfig.load("config/local.yaml"))
pipeline.build_index("data/input/10-K")            # or pipeline.load_index()

answer = pipeline.ask("Summarize liquidity risks for Tesla.", filters={"ticker": "TSLA"})
print(answer.text, answer.confidence, answer.abstained)
for c in answer.citations:
    print(c.filing_type, c.section_title, c.source_url)

report = pipeline.monitor("MSFT", "2023", "2024", forms=["10-K", "10-Q", "8-K"])
print(report.summary())

extraction = pipeline.extract("risk_factors", ticker="MSFT")
print(extraction.valid, extraction.data)
```

## 8. Config & environment

```bash
# Inspect/validate a config profile
python -c "from sec_llm_project.sec_intel.core.config import AppConfig; \
import json; print(json.dumps(AppConfig.load('config/local.yaml').to_dict(), indent=2))"
```

Override any setting via environment variables (dotted, double-underscore nested):

```bash
export SECI_LLM__PROVIDER=ollama
export SECI_LLM__MODEL=gemma3:12b
export SECI_RETRIEVAL__RERANK=true
export SECI_RETRIEVAL__TOP_K=10
sec-intel-mock demo                  # picks up SECI_* overlays on top of any --config
```

Profiles in `config/`:

| File | LLM | Embeddings | Notes |
| --- | --- | --- | --- |
| `local.yaml` | Ollama `gemma3:12b` | Nomic `nomic-embed-text` | **primary** — local/free; needs Ollama + `.[embeddings,vectorstore,yaml]` (see §1b) |
| `offline.yaml` / `offline.json` | mock | hashing | zero-dependency; tests/CI/demo |

---

## 9. Verified citations & faithfulness harness

Citation verification is **off by default** so the offline demo and CI require no keys.
Enable it for production use via environment variable or config.

### Enable it (local, independent judge)

The judge must be a **different local model** from the generator — a model can't
independently grade its own output (enforced by a self-grading guard). Generator
is `gemma3:12b`, so the judge is `mistral:7b`:

```bash
ollama pull mistral:7b
export SECI_VERIFICATION__ENABLED=true
export SECI_VERIFICATION__JUDGE__PROVIDER=ollama
export SECI_VERIFICATION__JUDGE__MODEL=mistral:7b

sec-intel --config config/local.yaml ask \
  "What liquidity risks did Tesla disclose?" --ticker TSLA
# answer.extra["verification"] contains: groundedness, supported/partial/unsupported/contradicted counts
```

### Tune the gate

```bash
# Lower the groundedness floor (default 0.6) — answers above the floor pass the gate.
export SECI_VERIFICATION__GROUNDEDNESS_FLOOR=0.7

# Disable the one-shot repair pass (keep only the original answer or abstain).
export SECI_VERIFICATION__REPAIR=false
```

### Offline / CI behaviour

When `SECI_VERIFICATION__JUDGE__PROVIDER` is `mock` or `hashing` (the CI defaults),
the verifier automatically uses `LexicalJudge` — a deterministic keyword-overlap judge
that needs no keys and no network. The evaluation suite always uses `LexicalJudge`
regardless of the production judge config.

```bash
# The evaluation always measures faithfulness (LexicalJudge fallback):
sec-intel-mock evaluate --demo
# Output includes: mean_groundedness, fully_grounded_rate, contradiction_rate
```

### Snapshot regression tracking

Aggregate metrics (excluding latency, which is environment-dependent) are compared to
`tests/eval/baseline.json`. Regressions are printed; **the exit code is never changed**
— the snapshot is for human review, not a CI gate.

```bash
sec-intel-mock evaluate --demo --snapshot              # diff vs baseline, print deltas
sec-intel-mock evaluate --demo --update-baseline       # rewrite baseline after intentional change
sec-intel-mock evaluate --demo --snapshot tests/eval/my_baseline.json  # custom path
```

### Python API

```python
from sec_llm_project.sec_intel.core.config import AppConfig, VerificationConfig, LLMConfig

cfg = AppConfig.load("config/local.yaml")
cfg.verification.enabled = True
cfg.verification.judge = LLMConfig(provider="ollama", model="mistral:7b")
cfg.verification.groundedness_floor = 0.7

from sec_llm_project.sec_intel.pipeline import SECIntelPipeline
pipeline = SECIntelPipeline(cfg)
pipeline.build_index("data/input/10-K")

answer = pipeline.ask("Summarize Tesla's liquidity risks.", filters={"ticker": "TSLA"})
vr = answer.extra.get("verification")   # VerificationReport.to_dict()
print(vr["groundedness"], vr["supported"], vr["contradicted"])
for c in answer.citations:
    print(c.verdict, c.support_quote)   # SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED
```
