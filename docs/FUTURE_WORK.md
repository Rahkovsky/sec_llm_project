# Future work / known limitations

Tracked follow-ups that are deliberately out of scope for now. Each entry notes
the current behavior, the root cause, and a proposed direction so the work can be
picked up cold.

---

## 1. Delisted / renamed tickers are skipped during EDGAR ingestion

**Status:** open · **Area:** `src/sec_llm_project/download/core.py` (`SECExtractor._get_filing`)

### Symptom

`sec-intel fetch` logs an error and skips a company whose ticker can no longer be
resolved, e.g.:

```
ERROR - Failed to fetch filing data for HES: Company not found: 'HES'
  Similar: 'HESM' (Hess Midstream LP), 'HSAI' (Hesai Group), 'HSIGF' (Hesai Group)
  Tip: Search by name with find_company("...") or pass a CIK directly.
```

The download does **not** crash — the failure is caught, logged, marked
`"missing"` in the fetch checkpoint, and the loop continues to the next ticker.
But the company's filings are never downloaded.

### Root cause

`edgartools` resolves filings by **current** ticker → CIK. When a company is
acquired, merged, or renamed, its ticker is delisted and dropped from that map,
so the lookup fails even though the historical filings still exist in EDGAR under
a stable **CIK**. (`HES` = Hess Corporation, acquired by Chevron and delisted in
2025; the bundled S&P 500 list is a Q1-2025 snapshot that still contains it.)

A delisted ticker is also indistinguishable from a transient lookup failure: both
get marked `"missing"` and are skipped on resume unless `fetch --reset` is passed.

### Proposed fix

Add a **CIK fallback** in `_get_filing`: when ticker resolution fails, retry via
`Company(cik)` (filings are keyed on the stable CIK) or `find_company(name)`
before giving up. Optionally maintain a small `ticker → CIK` override map for
known delisted S&P 500 constituents, and refresh the bundled constituent list.

Consider distinguishing "confirmed not found" from "transient error" in the
checkpoint so only the former is permanently skipped.

### References

- edgartools error hint: *"Search by name with `find_company("...")` or pass a CIK directly."*
- Hess Corporation CIK: `0000004447`

---

## 2. Exact-substring chunk fidelity for stricter verbatim citation spans

**Status:** open · **Area:** `src/sec_llm_project/sec_intel/chunking/sec_sections.py`,
`src/sec_llm_project/sec_intel/generation/verifier.py`

The citation verifier enforces that a judge's supporting quote occurs verbatim in
the cited chunk, comparing on whitespace-normalized text. Because the chunker
strips/normalizes whitespace and can merge a trailing window, a chunk's text is
not always a byte-exact substring of the original filing. A quote that spans a
window boundary can therefore fail the substring check even when faithful.

**Proposed direction:** store raw character offsets into the original filing text
on each chunk (the provenance sidecar already records section offsets) so the
verifier can validate against the source bytes rather than the normalized chunk.

---

## 3. BM25 / dense weight tuning — the global optimum is already covered

**Status:** resolved (analysis) · **Area:** `tools/tune_weights.py`,
`src/sec_llm_project/sec_intel/retrieval/fusion.py`

### Question

Do we need a more sophisticated optimizer (Bayesian search, gradient methods, a
2-D weight grid) to find the *globally* optimal BM25-vs-dense weighting?

### Answer: no — the existing 1-D sweep is provably complete for the weight

Both fusion methods depend **only on the ratio** of the two weights, not their
absolute magnitudes:

- **RRF** (`reciprocal_rank_fusion`): `fused = w_bm25·1/(k+rank) + w_dense·1/(k+rank)`.
  Scaling both weights by any constant `c` scales every fused score by `c`, so the
  ranking is unchanged.
- **Weighted fusion** (`weighted_fusion`): each score map is min-max normalized to
  [0,1], then combined as `w_dense·norm + w_bm25·norm`. Same argument — a common
  scale factor cancels out of the ordering.

Because only the ratio matters, constraining `dense_weight = 1 − bm25_weight` (the
simplex) **loses no generality**: the search space is genuinely one-dimensional.
A grid over a single bounded scalar **cannot be trapped in a local optimum** — it
evaluates the whole space — so the sweep *is* global optimization, not a
heuristic. A fancier optimizer would find nothing better; those tools exist for
high-dimensional or expensive search spaces, neither of which applies here. For
finer resolution use `--step 0.05` / `0.02`; the Recall/MRR curve is smooth.

### What is *not* yet globally tuned (the real remaining work)

The single weight is global, but three knobs that **interact** with it are held
fixed, so the joint optimum over `(fusion, weight, rrf_k, candidate_k)` is
unexplored:

| Parameter | Current | Why it couples with the weight |
| --- | --- | --- |
| `rrf_k` | 60 (fixed) | Sets how steeply rank position is discounted |
| `candidate_k` | 60 | Pool size before fusion — caps achievable recall |
| `fusion` | one per run | RRF vs weighted have different optima |

This is a small 2–3-D grid (~30 lines on top of the existing loop), not a new
optimizer.

### The deeper question a single global weight cannot answer

BM25 wins on exact-match queries (tickers, dollar figures, legal terms); dense
wins on paraphrased/semantic ones. A single global weight is a compromise across
that mix. A **per-category breakdown** of the sweep (Risk / Legal / Governance /
Company overview, available in the FinDER eval set) would reveal whether one
weight is leaving recall on the table — and if so, the answer is per-intent /
adaptive weighting, a larger design than any grid search.

### References

- Sweep tool and weight semantics: `tools/tune_weights.py`,
  `RetrievalConfig.bm25_weight` / `dense_weight` in `core/config.py`.
- Eval sets carry a `category` field for the per-category breakdown:
  `eval/finder_semantic.json`.

---

## 4. Groundedness ≠ correctness (deliberately deferred)

**Status:** recorded · postponed until the base eval architecture is solid ·
**Area:** `generation/verifier.py`, `evaluation/runner.py`

### The gap

Every automated faithfulness signal we have measures whether an answer is entailed
by the **retrieved** evidence — not whether it is **true**. A confidently wrong
answer that is faithfully grounded in a *wrongly-retrieved* chunk scores high on
groundedness. We observed exactly this: FinanceBench `mean_groundedness ≈ 0.87`
while `citation_correctness ≈ 0.03` — fluent, internally-consistent answers built
on the wrong evidence.

### Why it is hard

Correctness needs an external truth anchor (a gold answer or a human), and for the
numeric/financial questions that dominate FinanceBench it also needs reliable
table extraction and arithmetic checking. This is a substantially larger problem
than entailment scoring and couples retrieval quality, extraction, and reasoning.

### Decision

Defer the *solution*. First make the base architecture correct: real
(independent) judge enabled and validated against humans, retrieval weights tuned,
hermetic eval harness. Until then, **report groundedness and correctness as
separate axes and never let groundedness stand in for correctness** — the
[human-eval protocol](HUMAN_EVAL.md) scores them independently and tracks their
divergence, which is the metric to watch when this is picked up.

### Proposed direction (when resumed)

- Answer-correctness judge that compares the generated answer to the gold answer
  (FinDER / FinanceBench ship them), separate from the groundedness judge.
- For numeric questions, a structured numeric-equality check rather than text
  entailment. **Partially landed:** `generation/numeric.py` extracts figures
  (scale-/percent-aware) and `CitationVerifier` now downgrades a claim whose figure
  is absent from the cited chunk (numeric-hallucination guard). Known limit: it
  matches a figure present *anywhere* in the chunk, not the specific line item it
  belongs to — tightening that (figure ↔ label binding) is the remaining work.

---

## 5. Reward rich, correct, grounded answers — not cheap ones

**Status:** open (metric design) · **Area:** `evaluation/`, `generation/grounded.py`

### Problem

Overlap-based faithfulness can be *gamed by thin answers*. A short, vague response
that echoes a few evidence keywords can score as well as — or better than — a
precise, complete one. Optimising toward such a metric pushes the system toward
evasive one-liners. Confidence is also currently scaled by this groundedness
(`confidence *= 0.4 + 0.6·groundedness` in `grounded.py`), so a noisy lexical
signal propagates into the abstention gate.

### Proposed direction

- Add a **richness / completeness** axis (does the answer address all parts of the
  question, with specifics?) — already in the [human-eval rubric](HUMAN_EVAL.md);
  promote it to an automated signal (e.g. coverage of question sub-claims, or an
  LLM-judged completeness score) once the judge is validated.
- Define the reported quality as a **combination** of correct × grounded × rich,
  so a high score is only reachable by an answer that is simultaneously faithful,
  true, and complete — closing the "cheap score" loophole.
- Re-evaluate driving `confidence` off a validated judge signal rather than lexical
  overlap.

---

## 6. Claim decomposition + per-claim verification (judge robustness)

**Status:** open · **Area:** `generation/verifier.py`, `generation/prompts.py`

Today the judge receives the whole answer and is asked to break it into claims
itself. A stronger, less gameable design **decomposes the answer into atomic
claims first** (a dedicated step), then verifies each claim independently against
the evidence. Benefits: a half-true answer can't hide an unsupported clause inside
a mostly-supported sentence; groundedness becomes a clean per-claim average; and
each claim gets its own citation + verbatim quote + numeric check
(`numeric.figures_supported`). Pairs naturally with the verbatim-quote and
numeric invariants already in `CitationVerifier`. Consider a separate decomposition
model/prompt and caching, since it adds an LLM call per answer.

---

## 7. Question-answering generation (QAG) faithfulness

**Status:** open · **Area:** `evaluation/`, `generation/verifier.py`

QAG measures faithfulness by **round-tripping through questions**: generate
questions from the answer, answer them *using only the retrieved evidence*, and
check the two answers agree. Disagreement localizes exactly which assertion is
unsupported, and it is harder to game than overlap or single-pass entailment
because each sub-fact must independently survive evidence-only re-answering. It is
also a natural fit for the numeric checker (the re-answers are often figures).
Cost is the main caveat (several LLM calls per answer), so scope it to the eval
harness / spot checks first, not the live answer path.
