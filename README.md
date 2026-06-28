# SEC Disclosure Intelligence Prototype

A prototype for **retrieval, structured extraction, disclosure-change monitoring,
and citation-grounded answers over official SEC filings**. It is designed in the
spirit of regulatory analytics workflows (e.g. SEC DERA / OAAA–style disclosure
review): every answer is grounded in cited filing text, the system refuses when
evidence is weak, and all retrieval/generation events are auditable.

> This is a research/interview prototype. It is **not** an investment-advice tool
> and makes **no** prediction of wrongdoing. Where public SEC enforcement cases
> are used, they serve as an *evaluation* benchmark for retrieval quality — not as
> training labels for a misconduct detector.

## Highlights

- **Configurable model backend** — OpenAI / Anthropic / local Ollama, selected by
  config, never by code edits.
- **Hybrid retrieval** — BM25 lexical + dense vectors + metadata filtering +
  reciprocal-rank fusion + optional cross-encoder reranking.
- **SEC-aware chunking** — splits on `Item N.` boundaries; preserves ticker,
  company, form, date, fiscal year, section title, and item number.
- **Disclosure Change & Risk Signal Monitor** (flagship) — changed risk-factor
  language, new litigation, liquidity/going-concern signals, all cited.
- **Citation-grounded structured JSON** — risk factors, litigation, MD&A,
  internal controls, related-party, liquidity.
- **Verified citations with an independent judge** — each claim in an answer is
  independently evaluated for groundedness by a separately-configured LLM
  (cross-vendor: generate with OpenAI, judge with Anthropic). Fabricated quotes
  are caught by a deterministic substring check. A gate+repair pass drops or
  rewrites unsupported claims before the answer is returned.
- **Faithfulness harness** — evaluation measures `mean_groundedness`,
  `fully_grounded_rate`, and `contradiction_rate` alongside retrieval metrics.
  Results are compared to a versioned baseline and regressions are flagged
  (report-only; never gates CI).
- **Responsible AI** — confidence estimation, refusal thresholds, provenance,
  deterministic prompts, and a JSONL audit log.
- **Runs fully offline** for tests/CI/demo (deterministic mock + hashing
  embeddings — no API keys, no downloads).

## Compliance-aware architecture

This prototype is built around constraints relevant to regulated/government use:

| Property | How it is implemented |
| --- | --- |
| **No training on user data by default** | The system only *retrieves and reasons over* filings; it never fine-tunes on user inputs. |
| **Configurable model backend** | `AppConfig` selects the LLM/embedding provider; swap via config or `SECI_*` env vars. |
| **Local/private mode** | `config/local.yaml` runs Ollama + Nomic with no data leaving the host. |
| **Audit logs** | Every retrieval/answer is appended to `logs/audit.jsonl` with prompt/model/citation metadata. |
| **Prompt & version tracking** | Deterministic prompt templates; embedding version recorded in the index sidecar and verified on load. |
| **Cited evidence only** | Answers attach only the excerpts the model relied on; structured outputs require verbatim `evidence`. |
| **Refusal when confidence is weak** | Configurable `min_evidence_score` / `abstain_below_confidence` thresholds. |
| **Independent citation verification** | Claims are judged by a separately-configured LLM (cross-vendor); fabricated quotes caught by deterministic substring check; one gate+repair pass before the answer is returned. Off by default; enabled via `SECI_VERIFICATION__ENABLED=true`. |
| **Official SEC sources only** | Ingestion uses SEC EDGAR with fair-access rate limiting (≤ 10 req/s) and a descriptive User-Agent. |

## Default model stack (U.S.-aligned)

The recommended configuration (`config/default.yaml`) uses a strong commercial
generator and U.S.-origin embeddings, with a private local fallback:

```yaml
llm:
  provider: openai
  model: gpt-4.1-mini
embedding:
  provider: openai
  model: text-embedding-3-large
local_fallback:
  llm:        { provider: ollama, model: gemma3:12b }
  embedding:  { provider: nomic,  model: nomic-ai/nomic-embed-text-v1.5 }
```

Config profiles shipped in `config/`:

| Profile | LLM | Embeddings | Use |
| --- | --- | --- | --- |
| `default.yaml` / `default.json` | OpenAI `gpt-4.1-mini` | OpenAI `text-embedding-3-large` | Recommended / demo quality |
| `local.yaml` | Ollama `gemma3:12b` | Nomic `nomic-embed-text` | Air-gapped / private |
| `anthropic.json` | Anthropic `claude-opus-4-8` | OpenAI `text-embedding-3-large` | Alt. commercial |
| `offline.yaml` / `offline.json` | mock | hashing | Tests / CI / zero-dependency demo |

## Government deployment options

```
Deployment modes:
  1. Local/offline prototype using Ollama (Gemma 3) + Nomic (config/local.yaml).
  2. Commercial API prototype using OpenAI/Anthropic (config/default.yaml).
  3. Federal cloud deployment path using Azure OpenAI Government or
     AWS Bedrock / GovCloud (set llm.base_url / embedding.base_url to the
     authorized endpoint and keep keys in the environment).
```

**Compliance note:** this repository does **not** claim FedRAMP authorization. It
is *designed to be compatible with a FedRAMP-authorized deployment environment*:
backends are endpoint-configurable, secrets are read from the environment, no
data is used for training, and all activity is audit-logged.

## Install

```bash
# Install uv (fast Python package manager — https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
source $HOME/.local/bin/env                        # add uv to PATH (current shell)
# brew install uv                                   # macOS Homebrew alternative (no PATH step needed)
# pip install uv                                    # any platform

# Install Python 3.13
uv python install 3.13                             # uv manages Python versions
# brew install python@3.13                          # macOS Homebrew alternative
# or download from https://python.org/downloads/

uv venv --python 3.13     # run AFTER uv python install 3.13 to avoid download prompts
source .venv/bin/activate

# Core platform only (stdlib — runs the offline demo, tests, evaluation):
uv pip install -e .

# Optional backends as needed:
uv pip install -e ".[openai]"        # OpenAI generation + embeddings
uv pip install -e ".[anthropic]"     # Anthropic generation
uv pip install -e ".[embeddings]"    # local Nomic embeddings (sentence-transformers)
uv pip install -e ".[vectorstore]"   # Chroma persistent vector store
uv pip install -e ".[sec]"           # EDGAR ingestion (edgartools)
uv pip install -e ".[dev,yaml]"      # tests + YAML config
```

Configure secrets and SEC identity in the environment (never in code):

```bash
export OPENAI_API_KEY=...            # for the default config
export SEC_USER_NAME="Your Name"     # used to build a compliant EDGAR User-Agent
export SEC_USER_EMAIL="you@org.gov"
```

## Quickstart

See **[docs/COMMANDS.md](docs/COMMANDS.md)** for the full, copy-pasteable test &
run reference (setup, every CLI subcommand, ingestion, Docker, Python API, env
overrides). For the **fully-local stack** (Ollama gemma3:12b + Nomic, no keys) —
model install, full-corpus indexing, and a live status log — see
**[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**.

```bash
# Fully offline, no keys: indexes a built-in synthetic corpus and answers a
# grounded question with citations.
sec-intel demo

# The flagship monitor over the synthetic corpus:
sec-intel monitor NOVA 2023 2024 --demo

# Retrieval + faithfulness metrics (LexicalJudge; no keys required):
sec-intel evaluate --demo

# Snapshot: compare metrics against the committed baseline in tests/eval/baseline.json
sec-intel evaluate --demo --snapshot

# Enforcement-case benchmark (evaluation, not training):
sec-intel evaluate --demo --enforcement
```

With real filings and the default (OpenAI) config:

```bash
# 1) Download official filings — multi-form, multi-year (rate-limited, compliant UA).
#    Two+ consecutive years are needed for the year-over-year monitor/compare.
sec-intel fetch MSFT TSLA --forms 10-K 10-Q 8-K --years 2023-2025

# 2) Build an index from a downloaded form directory
sec-intel --config config/default.yaml index data/input/10-K

# 3) Ask a grounded question (cited, with confidence + abstention)
sec-intel --config config/default.yaml ask \
  "Summarize liquidity risks for Tesla with citations." --ticker TSLA

# 4) Monitor disclosure changes between two years (the flagship)
sec-intel --config config/default.yaml monitor MSFT 2024 2025

# 5) Structured, citation-grounded extraction
sec-intel --config config/default.yaml extract risk_factors --ticker MSFT
```

## Flagship: Disclosure Change & Risk Signal Monitor

Input: a ticker, two fiscal years, and forms (`10-K`, `10-Q`, `8-K`). Output (all
citation-grounded):

- **changed risk-factor language** (added / removed / reworded Item 1A units),
- **new legal/litigation disclosures** (Item 3 additions),
- **liquidity / going-concern signals** (deterministic phrase lexicon, each with a
  source quote),
- **optional XBRL financial context** (SEC company-facts hook),
- a **summary** with counts and a going-concern flag.

```bash
sec-intel monitor NOVA 2023 2024 --demo | jq '.summary'
# { "new_risk_factors": 2, "new_litigation_disclosures": 1,
#   "risk_signals": 3, "going_concern": false }
```

## Evaluation (not training)

The evaluation suite measures the system, it does not learn from labels:

- **Retrieval:** Recall@k, MRR, MAP@k, latency (p95).
- **Grounding:** citation correctness, hallucination rate, abstention recall.
- **Faithfulness:** `mean_groundedness`, `fully_grounded_rate`, `contradiction_rate` —
  each non-abstained answer is scored by an independent `LexicalJudge` (offline, no keys)
  or the configured `LLMJudge` when verification is enabled.
- **Snapshot regression:** aggregate metrics are compared against a versioned baseline
  (`tests/eval/baseline.json`) and per-metric deltas printed. Regressions are flagged but
  **never gate CI** — the report is for human review.
- **Enforcement-case benchmark:** public SEC enforcement *categories*
  (e.g. revenue-recognition material weakness, going-concern omission,
  related-party self-dealing) are used as retrieval test cases. We measure
  whether the system **retrieves the relevant evidence and surfaces the known
  risk signals** an analyst would expect — framed explicitly as evaluation of
  retrieval/organization, with each case linking to a public SEC resource. To
  evaluate real entities, index their filings and provide cases via
  `--cases cases.json`.

```bash
sec-intel evaluate --demo                         # retrieval + faithfulness metrics
sec-intel evaluate --demo --snapshot              # diff vs committed baseline
sec-intel evaluate --demo --update-baseline       # rewrite baseline after intentional changes
sec-intel evaluate --demo --enforcement --json    # enforcement benchmark
```

## Architecture

```
src/sec_llm_project/sec_intel/
├── core/         # config (DI), shared types, audit/provenance
├── llm/          # provider abstraction: mock | ollama | openai | anthropic
├── embeddings/   # openai (default) | nomic (local) | hashing (offline), versioned
├── chunking/     # SEC-aware section splitting + provenance
├── index/        # chunk catalog + vector store (memory | chroma) + reproducible builder
├── retrieval/    # bm25 + dense + fusion (rrf/weighted) + rerank → HybridRetriever
├── extraction/   # JSON-schema-constrained structured extraction
├── generation/   # grounded answers: citations, confidence, abstention, citation verifier
├── comparison/   # filing-vs-filing diff engine
├── monitor/      # Disclosure Change & Risk Signal Monitor (flagship)
├── evaluation/   # metrics + curated set + enforcement benchmark
├── ingest/       # rate-limited EDGAR download (fair-access, UA)
├── app/cli.py    # sec-intel command-line demo
└── pipeline.py   # SECIntelPipeline facade wiring everything from one config
```

## Development

```bash
uv pip install -e ".[dev,yaml]"
pytest -q            # offline, deterministic
ruff check src tests
```

CI (`.github/workflows/ci.yml`) runs lint, the offline test suite, and the demo
on Python 3.13 with **no** ML/vector backends installed.

## Roadmap / known limitations

Tracked follow-ups (e.g. resilience for delisted/renamed tickers during EDGAR
ingestion) live in **[docs/FUTURE_WORK.md](docs/FUTURE_WORK.md)**.

## License

MIT
