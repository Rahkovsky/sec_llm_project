# Observability & QC — tracing with Arize Phoenix

Every `ask` is traced as a tree of spans so you can see *why* an answer came out
the way it did — which chunks were retrieved and their scores, what the model
produced, the judge's groundedness, and the latency of each stage. Two backends,
both keeping data **on your machine**:

- **`jsonl`** (default, zero extra deps) — nested spans appended to a file. Great
  for `grep`/`jq` and as an always-on record.
- **`phoenix`** — exports OpenTelemetry spans to a locally-run **Arize Phoenix**,
  a free open-source UI for traces, retrieval inspection, and evals.

Off by default. Nothing is captured, and no dependency is added, until you enable it.

---

## What a trace looks like

```
[CHAIN]     answer     input, evidence_score, output, confidence, abstained, n_citations
  [RETRIEVER] retrieve   n_retrieved, latency_ms, documents[{id, score, content}]
  [LLM]       generate   model, output
  [EVALUATOR] verify     groundedness, contradicted, supported, total
```

All of that is data the pipeline already computes — the tracer just structures it.

---

## Option A — JSONL (no install, start here)

Enable it in `config/local.yaml` (already scaffolded):
```
observability:
  enabled: true
  backend: jsonl
  path: logs/traces.jsonl
```
or per-run without editing the file:
```
export SECI_OBSERVABILITY__ENABLED=true
sec-intel --config config/local.yaml ask "What are Apple's risk factors?" --ticker AAPL
```

Inspect with `jq` — e.g. the slowest stage per request, or every abstention:
```
jq -c 'select(.name=="retrieve") | {ms, n:.attributes.n_retrieved}' logs/traces.jsonl
jq -c 'select(.attributes.abstained==true) | {q:.attributes.input, why:.attributes.abstain_reason}' logs/traces.jsonl
```

---

## Option B — Phoenix (the UI)

**1. Install the exporter** (our side — the OTLP client):
```
uv pip install -e ".[observability]"
```

**2. Run Phoenix** (the server — separate, free, local). Either:
```
uv pip install arize-phoenix
phoenix serve
```
or Docker:
```
docker run -p 6006:6006 arizephoenix/phoenix:latest
```
Phoenix comes up at **http://localhost:6006**; its OTLP collector listens on
`http://localhost:6006/v1/traces`.

**3. Point the app at it** — flip the backend:
```
observability:
  enabled: true
  backend: phoenix
  endpoint: http://localhost:6006/v1/traces
```
or:
```
export SECI_OBSERVABILITY__ENABLED=true
export SECI_OBSERVABILITY__BACKEND=phoenix
```

**4. Ask something, then open the UI:**
```
sec-intel --config config/local.yaml ask "What legal proceedings does Microsoft disclose?" --ticker MSFT
```
Each request appears as a trace. Because spans carry OpenInference attributes,
Phoenix renders the `retrieve` span as a **retriever** (expand it to see the
retrieved chunks and their fused scores) and `generate` as an **LLM** span with
its output — so you can click straight from a bad answer to the evidence it used.

---

## Using it for QC / evals

Phoenix's eval features are free and self-hosted, which is what makes it a good
fit here:

- **Retrieval QC.** Open a `retrieve` span and check whether the right chunk was
  actually in the pool — the fastest way to tell a retrieval miss from a
  generation miss (the exact `groundedness` ≠ `correctness` gap in
  [FUTURE_WORK.md](FUTURE_WORK.md) §4).
- **Batch evals.** Send the FinDER / FinanceBench sets through and let Phoenix run
  its LLM-as-judge templates (hallucination, QA-correctness, relevance) over the
  captured traces, then compare runs. This complements the built-in harness
  (`sec-intel evaluate`) and the [human-eval protocol](HUMAN_EVAL.md) — same data,
  a UI on top.

---

## Notes

- **Data stays local.** Both backends write to your host (a file, or a Phoenix you
  run yourself). Do not point the endpoint at a hosted collector for filing data.
- **Zero-cost when off.** With `enabled: false` the tracer is a no-op and no
  OpenTelemetry import happens.
- **Token/cost** capture is not wired yet — it needs the Ollama provider to
  surface usage counts (it returns them in the response). Small follow-up.
- The `phoenix` backend needs the `observability` extra; the `jsonl` backend does
  not. Running Phoenix itself is independent of this package.

See also: [LOCAL_SETUP.md](LOCAL_SETUP.md), [HUMAN_EVAL.md](HUMAN_EVAL.md).
