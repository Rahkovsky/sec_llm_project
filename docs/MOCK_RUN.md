# 2. Mock Run — `sec-intel-mock` (NOT a real model)

> **This is not the real system.** `sec-intel-mock` runs a **deterministic mock
> LLM** over a **synthetic corpus** — no Ollama, no real embeddings, no filings.
> It exists only to exercise the retrieval/grounding/evaluation *plumbing* with
> zero dependencies (tests, CI, quick smoke checks). A mock answer looks fluent
> and cited but is **not** a real answer and must never be read as one.

For real answers over real filings, see **[1. Installation](INSTALLATION.md)** →
**[3. Real Run](REAL_RUN.md)**.

---

## What the mock actually is

`sec-intel-mock` is a **separate binary** from `sec-intel`. It forces the explicit
offline stack (`AppConfig.offline()`): mock LLM + hashing embeddings + in-memory
index. Two pieces combine:

1. **A deterministic mock LLM** (`model: mock-1`) — assembles an extractive
   "answer" from the retrieved text. Not a real model; not indicative of answer
   quality.
2. **A built-in synthetic corpus** — a fictional company, **"Nova Devices"**
   (~15 chunks). Not your SEC filings, not the Chroma index.

So a mock answer is: **mock generator + hashing embeddings + real BM25/dense
retrieval over a fake 15-chunk corpus.** The retrieval/citation machinery is
genuine; the model and the data are not.

## The real binary never falls back to this

`sec-intel` (the real binary) requires `--config config/local.yaml` and a real
model install. Its model-using commands **refuse to run** without a config rather
than silently producing mock output. You can only reach the mock by deliberately
running `sec-intel-mock`.

## Requirements

Only the stdlib core — no ML/vector backends:

```bash
uv pip install -e .        # see docs/INSTALLATION.md §1a
```

## Commands

```bash
# End-to-end demo: index the synthetic corpus, answer a sample question.
sec-intel-mock demo
sec-intel-mock demo --question "What cybersecurity risks does Nova Devices face?"

# Flagship disclosure-change monitor over the synthetic corpus:
sec-intel-mock monitor NOVA 2023 2024 --demo | jq '.summary'
# { "new_risk_factors": 2, "new_litigation_disclosures": 1,
#   "risk_signals": 3, "going_concern": false }

# Retrieval + faithfulness metrics (offline LexicalJudge; no keys):
sec-intel-mock evaluate --demo
sec-intel-mock evaluate --demo --snapshot              # diff vs committed baseline
sec-intel-mock evaluate --demo --update-baseline       # rewrite the baseline
sec-intel-mock evaluate --demo --enforcement --json    # enforcement-case benchmark
```

These are exactly what CI runs — the offline suite executes on Python 3.13 with
**no** ML/vector backends installed, which is why the mock path stays
dependency-free and deterministic.

## How to tell you're in mock mode

The answer footer prints the model. `sec-intel-mock` always shows:

```
Confidence: 0.93  (model: mock-1)
```

If you see `model: mock-1`, you ran the mock. A real run via
`sec-intel --config config/local.yaml` prints `model: gemma3:12b`.

> **Rule of thumb:** mock output only ever comes from the `sec-intel-mock` binary
> (or an explicit `--config config/offline.yaml`). If you didn't ask for the mock,
> you won't get it.
