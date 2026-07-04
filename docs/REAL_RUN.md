# 3. Real Run — real model over your filings

This is the actual system: local **gemma3:12b** + **Nomic** embeddings answering
over the SEC filings you indexed. Everything runs on-device.

**Prerequisite:** complete **[1. Installation](INSTALLATION.md)** first (packages,
model, filings, and a built index). Confirm before you start:

```bash
source .venv/bin/activate                 # every new terminal
ollama list                               # gemma3:12b present
curl -s http://localhost:11434/api/version   # Ollama server up
ls data/sec_index/chroma_db               # index exists
```

> All real commands take `--config config/local.yaml`. **Use the same config for
> indexing and querying** or you'll get an `Embedding mismatch` error. If a
> model-using command is run **without** `--config`, `sec-intel` refuses (it will
> not silently fall back to the mock — that's `sec-intel-mock`, see
> [2. Mock Run](MOCK_RUN.md)).

---

## Ask a grounded question

```bash
sec-intel --config config/local.yaml ask \
  "What are Apple's most significant risk factors?" --ticker AAPL

sec-intel --config config/local.yaml ask \
  "Summarize liquidity risks for Tesla with citations." --ticker TSLA
```

Answers are cited, carry a confidence score, and **abstain** when evidence is
weak. Flags: `--ticker` scopes to one company, `--top-k N` widens the evidence
pool, `--json` emits machine-readable output (pipe to `jq`).

## Flagship: Disclosure Change & Risk Signal Monitor

Year-over-year changes for a ticker — changed risk-factor language, new
litigation, liquidity/going-concern signals, all citation-grounded:

```bash
sec-intel --config config/local.yaml monitor MSFT 2024 2025
sec-intel --config config/local.yaml monitor AAPL 2024 2025 | jq '.summary'
```

## Structured extraction

Citation-grounded JSON (risk_factors, litigation, mdna, internal_controls,
related_party, liquidity):

```bash
sec-intel --config config/local.yaml extract risk_factors --ticker MSFT
```

## Compare two years

```bash
sec-intel --config config/local.yaml compare AAPL 2024 2025 --items 1A 3 7
```

## Evaluate over the real index

```bash
sec-intel --config config/local.yaml evaluate --load-index
```

(`--load-index` evaluates against the persisted full index rather than rebuilding.)

---

## Notes / gotchas

- **Latency:** the first question after idle adds a one-time ~5 s model load; then
  answers land in ~3–10 s (longer for risk-factor questions that pull many large
  chunks into context). Set `keep_alive: -1` to keep the model resident.
- **Don't double-run `ollama serve`** — the macOS app already runs the server; a
  second one errors with "address already in use".
- **Switching models:** change `llm.model` in `config/local.yaml`, or override
  with `export SECI_LLM__MODEL=...`.
- **Independent citation verification** (a *different* local model grades each
  claim) is off by default; enable with `SECI_VERIFICATION__ENABLED=true` after
  `ollama pull mistral:7b`.
- **zsh + inline comments:** never paste a command with a trailing `# comment` —
  zsh without `interactive_comments` passes it as arguments.

See also: [COMMANDS.md](COMMANDS.md) (full CLI reference),
[OBSERVABILITY.md](OBSERVABILITY.md) (tracing / Phoenix for QC).
