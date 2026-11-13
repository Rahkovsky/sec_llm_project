# Experiment log: retrieval and answer quality (Nov 2026)

What we tried, how it was measured, and what it did to the metrics. Newest
results are at the end of each section; open items are in the last section.
All numbers are reproducible from the tools named in each section.

## 0. Setup and definitions

**System.** Local stack on an M1 Pro / 16 GB (`config/local_16gb.yaml`):
generator `gemma4:12b`, judge `qwen3:14b` (Ollama), embedder
`Qwen/Qwen3-Embedding-0.6B`, Chroma index. Retrieval is scoped to the
question's company by a ticker filter and returns the top 8 chunks.

**Corpus.** 470 tickers; per company three 10-Ks (filed 2023-2025, FY2022-2024),
about three 10-Qs (one per year) and 8-Ks. Index `data/sec_index_qwen06`.

**Evaluation sets.**

| Set | Size | Use |
|---|---|---|
| FinDER gold (`eval/finder_semantic.json`) | 260 questions with gold passages | Retrieval and end-to-end |
| FinDER dev / test split (`eval/finder_split.json`) | 128 / 132 | Tune on dev, confirm once on test |
| Placebos (in `eval/gold_plus_placebo.json`) | 6 unanswerable questions | Must be refused |
| Claude-written synthetic Q/A (`eval/synth_claude.json`) | 361 | Retrieval diagnosis only |

**Retrieval metrics.**
- *Relevant chunk*: passes `passage_hit` against FinDER's gold passage: it
  holds ≥50% of the passage's tokens, or lies ≥80% inside it (≥40 tokens).
- *Scorable*: at least one chunk of the company is relevant (dev 122 of 128,
  test 126 of 132).
- *Hit@8*: share of scorable questions with a relevant chunk in the top 8.
  *MRR@8*: mean reciprocal rank of the first relevant chunk (0 if not in top 8).
- Comparisons are *paired*: Δ vs a baseline on the same questions, with a 95%
  bootstrap CI (5,000 resamples) and wins/losses (questions only one side hits).

**End-to-end metrics** (`tools/provenance_dag.py` → `tools/provenance_report.py`,
266 questions = 260 FinDER + 6 placebos).
- *Faithfulness*: share of answer claims the verifier marks supported.
- *Context recall, strict / cover*: share of the gold passage covered by the
  retrieved chunks, by the legacy rule / by the "cover" rule above.
- *Weighted claim error rate*: unsupported or contradicted claims, weighted.
- *Hardened qualifiers*: claims that drop the filing's hedging ("may" → "will").
- *Error sources*: each question assigned one dominant failure class.

## 1. Embedder regression: Qwen vs Nomic (Nov 10)

**Question.** After switching from Nomic to Qwen3-Embedding, retrieval looked
worse. Why?

**Method.** Same chunks, both embedders, 40 dev tickers (`tools/ab_embedders.py`,
`tools/ab_diagnose.py`, `out/ab_diag/report.txt`), FinDER questions plus the
Claude-written synthetic questions.

**Ruled out.** Index corruption, Chroma approximate search, truncation,
EOS/padding, chunk length, score bunching (Qwen's score spread is 2× Nomic's).
bf16 query encoding adds minor noise; fixed by encoding queries in fp32.

**Cause: the query instruction.** Qwen embeds queries with an instruction
prefix. The original one ("instr0": *"Given a question about SEC filings,
retrieve the passages that answer it"*) was the worst variant on every query
set. It pulled filing boilerplate into the top 8: 34.5% of results vs 19% of
the corpus (24.8% with no instruction).

**Instruction variants, all 122 scorable FinDER dev questions** (`tools/ab_finder_dev.py`,
`out/ab_diag/finder_dev_full.txt`):

| Variant | Hit@8 | MRR@8 | Δ vs instr1 [95% CI] | W/L |
|---|---|---|---|---|
| **instr1** ("terse search phrase from a financial analyst … 10-K annual report") | **0.525** | 0.321 | — | — |
| web (model-card default) | 0.492 | 0.332 | −0.033 [−0.074, 0.000] | 1/5 |
| no instruction | 0.393 | 0.297 | −0.131 [−0.205, −0.066] | 3/19 |
| instr0 (original) | 0.361 | 0.237 | −0.164 [−0.238, −0.098] | 1/21 |
| BM25 alone | 0.311 | 0.238 | −0.213 [−0.303, −0.123] | 5/31 |

On the 46 questions where Nomic vectors were cached: instr1 0.522, Nomic 0.413.

**Decision.** instr1 adopted in `embeddings/huggingface.py`; queries in fp32.

## 2. Hybrid BM25 + dense (Nov 10)

**Method.** Weighted reciprocal-rank fusion (k=60) of the dense and BM25 top-60
lists, BM25 using the global corpus statistics (`tools/ab_finder_dev.py`).

| instr1 + BM25 weight | Hit@8 | MRR@8 | Δ vs dense only [95% CI] | W/L |
|---|---|---|---|---|
| 0.0 (dense only) | **0.525** | 0.321 | — | — |
| 0.3 | 0.451 | 0.353 | −0.074 [−0.139, −0.008] | 4/13 |
| 0.5 | 0.434 | 0.334 | −0.090 [−0.164, −0.016] | 5/16 |
| 0.7 | 0.402 | 0.326 | −0.123 [−0.197, −0.041] | 5/20 |

BM25 raises MRR slightly but lowers Hit@8 on FinDER's terse questions. On the
46 questions also rewritten as natural questions, BM25 0.3 helped
(0.609 → 0.652, CI includes 0). An earlier 5-fold CV over BM25 weights
(`tools/cv_weights.py`, old instruction) also picked 0.0-0.1.

**Decision.** BM25 off (`use_bm25: false`); it also saves ~5 GB RAM and ~2 s/query.

## 3. End-to-end: v1 → v2 (Nov 11)

v1 = `out/provenance_16gb` (original instr0 query instruction).
v2 = `out/provenance_16gb_v2` (instr1 instruction, fp32 queries). Both
`config/local_16gb.yaml`, 266 questions.

| Metric | v1 | v2 |
|---|---|---|
| Answered | 82 | 115 |
| Faithfulness | 0.763 | 0.845 |
| Context recall, strict / cover | 0.146 / 0.300 | 0.235 / 0.473 |
| Weighted claim error rate | 0.474 | 0.311 |
| Hardened-qualifier claims | 28 | 39 |
| Placebos refused | 6/6 | 6/6 |
| Wrong-period answers (year-named questions) | 1/2 | 1/3 |

## 4. Why v2 still fails: error analysis (Nov 11)

Per-question data: `out/provenance_16gb_v2/analysis_rows.csv`, `rank_rows.csv`,
`rank_items.csv`.

| Class | Questions | Evidence fully in index |
|---|---|---|
| A. Retrieval missed → abstained | 97 | 76 |
| B. Gold retrieved → abstained anyway | 48 | 44 |
| C. Retrieval missed → answered from other text | 22 | 18 |
| D. Gold retrieved → unsupported claims | 29 | 28 |
| E. Clean except dropped qualifiers | 10 | 10 |
| F. Clean | 54 | |

**4.1 Retrieval misses are deep.** Dense rank of the gold chunk on the 87
measurable class-A misses: 1-8: 2, 9-20: 12, 21-60: 18, >60: 55. Overall
Hit@20 = 0.597, so a larger top-k or a reranker over the top 60 cannot fix most
of them. Misses carry more boilerplate (3.5 cover/exhibit chunks per top 8 vs
2.2 for clean answers) and more 10-Q/8-K chunks.

**4.2 Offline retrieval what-ifs** (211 scorable, Hit@8):

| Change | Hit@8 |
|---|---|
| Production | 0.524 |
| Remove near-duplicates + chunks labelled item 0 | 0.581 |
| Oracle section routing (search only the gold passage's own item) | 0.742 |

The 0.581 turned out to come from the item-0 filter, not deduplication (§6),
and that filter wrongly drops most 10-Q chunks (§4.6). Oracle routing is an
upper bound, not a method.

**4.3 All-or-nothing abstention.** Most abstentions are the generator's own
`INSUFFICIENT_EVIDENCE` (A: 85 bare + 6 partial; B: 40 bare + 4 partial).
`generation/grounded.py:104` abstains when the flag appears *anywhere*, so ~10
substantive answers with one flagged sub-part were discarded (e.g. finder_0042,
IRM). The 40 bare B cases ask for inferences the filings don't state
("investor confidence", "stock performance").

**4.4 Verifier false negatives.** Of 126 claims marked not supported:

| Cause | Claims |
|---|---|
| Correct quote rejected only for punctuation | 37 |
| Correct quote rejected only for curly quotes / dashes | 20 |
| Quote is in a different retrieved chunk (wrong citation number) | 9 |
| "The excerpts do not contain…" meta-statement scored as a claim | 11 |
| No quote given | 18 |
| Quote not found (paraphrase or invented) | 41 |
| Numeric check failed | 1 |

Only 1 unsupported claim is truly absent from the retrieved text; 55 have ≥90%
of their tokens present. Cause: `CitationVerifier._quote_in_source` normalizes
whitespace only.

*Offline re-score of v2* (punctuation-insensitive word-level quote match in any
retrieved chunk, quotes ≥5 words, meta-claims excluded): claim support
0.845 → 0.925 (D 0.765 → 0.910, C 0.807 → 0.912); 24 of 51 C+D questions
become fully supported; 6 of 10 groundedness-floor abstentions would pass 0.6.
This is an upper bound: the judge's verdict before the deterministic quote
check was not stored.

**4.5 Smaller issues.** 39 hardened-qualifier claims in 10 questions; 1 of 3
year-named questions answered from the wrong period.

**4.6 10-Q section labels.** In 88% of 10-Qs more than half the chunks are
labelled item "0" (Cover / Front Matter): the segmenter targets 10-K item
headings, and 10-Q items (Part I Items 1-4, Part II Items 1-6) are not
recognized (hypothesis; not yet confirmed in code). Also 10% of 8-Ks, 1% of 10-Ks.

## 5. Corpus coverage and entity resolution (Nov 11)

**5.1 Evidence audit** (`tools/evidence_audit.py` → `out/evidence_audit.csv`):
share of the gold passage present in the best single filing.

| Coverage | Dev | Test | All |
|---|---|---|---|
| ≥90% | 111 | 113 | 224 |
| 50-90% | 11 | 13 | 24 |
| None | 6 | 6 | 12 |

Best-matching filing: 10-K 242, 10-Q 6; FY2022 23, FY2023 164, FY2024 57,
FY2025 4. Of the 12 with no evidence: 4 FinDER labelling errors (question names
another company: 0159 CE, 0157 DHR, 0086 IP, 0145 IT), 2 MO (download bug),
6 unexplained (MCD 0231, AMP 0098, ADI 0125, MTCH 0135, MA 0209, IT 0212).

**5.2 Download bugs found.** edgartools' `get_filings(form="10-K")` includes
amendments, and the downloader took the first filing per year: 26 of 1,402
10-K files were 10-K/A (MO and ALB for all three years). BK, WFC, IBM and USB
10-Ks are Exhibit 13 wrappers (~100k characters; the annual report is an
attachment). Fixed in `download/core.py` (amendments excluded; EX-13 text
appended for 10-Ks under 150k characters). Re-download: `tools/refetch_10k.sh`.

**5.3 Entity resolver** (`retrieval/entity.py`, `tools/eval_entity.py`):
deterministic question → ticker, replacing the oracle ticker from the eval
case. Exact 258/265 (97.4%), main company correct 261/265 (98.5%); the 4
misses are the FinDER label errors above, where the resolver follows the
question text. The placebo about SpaceX resolves to nothing. Not yet used in an
end-to-end run.

**5.4 v3 corpus** (`data/sec_index_v3`, built Nov 11). 10-K/A and Exhibit-13-wrapper
10-Ks replaced by the real annual report (26 filings removed), plus 10-Ks filed in 2022
(fiscal 2021). Only the 205 new 10-Ks of the 200 FinDER/synth tickers are embedded so
far (fp16 document embeddings; cos to the stored bf16 vectors ≥ 0.9998); the 281 new
10-Ks of other tickers are pending and cannot affect any ticker-filtered metric.
`data/sec_index_v3_gold` = v3 + every FinDER gold passage as its own pseudo-filing
(730,554 chunks).

Evidence audit on v3 (`out/evidence_audit_v3.txt`): gold ≥ 90% present 224 → 227,
none 12 → 11. No question's best-matching filing is fiscal 2021, so the new 2022-filed
10-Ks add no FinDER evidence; they matter for the corpus, not this benchmark.

## 6. Diversifying the top 8 (Nov 11)

**Question.** Can a form- and section-agnostic re-ordering of the dense
candidates lift Hit@8? (`tools/diversity_lab.py`; results
`out/diversity/sec_index_qwen06_{dev,test}.txt`.)

**Variants** (all reorder the same dense instr1 candidates, using only chunk
text or embeddings):

| Variant | Rule |
|---|---|
| base | Dense top 8 (production) |
| dedup[best] / dedup[new] | Collapse near-duplicates (token Jaccard ≥0.8) in the top 200; keep the best-scoring / most recently filed copy |
| mmr@λ | After dedup, maximal marginal relevance over the top 60: each pick maximizes λ·cos(q,d) − (1−λ)·max cos(d, picked) |
| cap@N | After dedup, at most N chunks per filing |
| clus@P | After dedup, k-means (8) over the top P; best chunk per cluster |

**Dev (122 scorable)** — λ chosen here. Dense rank of gold: 1-8: 64, 9-20: 9,
21-60: 15, 61-200: 15, >200: 19.

| Variant | Hit@8 | MRR@8 | Δ [95% CI] | W/L | Distinct filings / sections in top 8 |
|---|---|---|---|---|---|
| base | 0.525 | 0.312 | — | — | 4.6 / 2.7 |
| mmr@0.5 | 0.598 | 0.342 | +0.074 [−0.008, +0.156] | 18/9 | 4.4 / 4.8 |
| mmr@0.7 | 0.590 | 0.346 | +0.066 [0.000, +0.131] | 12/4 | 4.4 / 4.5 |
| **mmr@0.85** | **0.590** | **0.354** | **+0.066 [+0.016, +0.123]** | **10/2** | 4.4 / 4.1 |
| clus@60 | 0.566 | 0.344 | +0.041 [−0.025, +0.115] | 12/7 | 4.7 / 4.4 |
| cap@3 | 0.549 | 0.335 | +0.025 [0.000, +0.057] | 3/0 | 4.7 / 3.2 |
| cap@2 | 0.541 | 0.333 | +0.016 [−0.025, +0.057] | 4/2 | 5.2 / 3.4 |
| dedup[best] | 0.533 | 0.332 | +0.008 [0.000, +0.025] | 1/0 | 4.4 / 3.1 |
| dedup[new] | 0.533 | 0.332 | +0.008 [0.000, +0.025] | 1/0 | 4.4 / 3.1 |
| clus@200 | 0.434 | 0.312 | −0.090 [−0.172, −0.016] | 6/17 | 4.5 / 4.4 |

**Test (126 scorable)** — run once, after choosing on dev.

| Variant | Hit@8 | MRR@8 | Δ [95% CI] | W/L |
|---|---|---|---|---|
| base | 0.524 | 0.367 | — | — |
| **mmr@0.85** | **0.603** | **0.386** | **+0.079 [+0.032, +0.135]** | **11/1** |
| mmr@0.5 | 0.556 | 0.376 | +0.032 [−0.056, +0.119] | 17/13 |
| mmr@0.7 | 0.548 | 0.392 | +0.024 [−0.048, +0.095] | 12/9 |
| dedup, cap@2, cap@3, clus@60 | 0.540 | ~0.373 | +0.016 | 2-10 / 0-8 |
| clus@200 | 0.476 | 0.349 | −0.048 [−0.127, +0.040] | 11/17 |

**Findings.**
- MMR λ=0.85 is the only variant with a CI excluding zero on both splits, with
  very few losses. It rescues gold ranked 9-60 and nothing deeper.
- Exact near-duplicate collapse does almost nothing: year-to-year versions of a
  section are chunked at different boundaries, so whole-chunk overlap rarely
  reaches 0.8. The redundancy is same-*topic*, which embedding similarity catches.
- Clustering a deep pool (200) hurts: one chunk per topic forces irrelevant topics in.
- Lower λ (more diversity) wins more on dev but loses more; it did not hold on test.

**Decision.** `retrieval.mmr_lambda: 0.85`, `mmr_pool: 60` in
`config/local_16gb*.yaml`; implemented in `retrieval/hybrid.py`
(`HybridRetriever._mmr`, using stored vectors via `SECIndex.vectors`).
Test: `tests/test_retrieval.py::test_mmr_skips_redundant_duplicate`.

**Production check** (dev, the real `HybridRetriever` with the config above vs
the lab's per-question results): base Hit@8 0.516 (lab 0.525; agrees on 121/122),
MMR 0.598 (lab 0.590; agrees on 119/122). The small differences most likely come from
production's approximate Chroma search (HNSW) over a 60-candidate pool vs the
lab's exact search; the gain is the same (+0.08).

## 7. Query rewriting (Nov 11)

**Question.** About a quarter of scorable questions have their gold chunk
ranked below 60 by dense retrieval (dev 34, test 37), out of reach for MMR.
Does rewriting FinDER's terse analyst phrases with the local generator help?
(`tools/rewrite_lab.py`; rewrites cached in `out/rewrite/rewrites_gemma4_12b.json`.)

**Rewrites** (gemma4:12b, temperature 0, three calls per question, no access
to the gold passage):

| Rewrite | Prompt (abridged) |
|---|---|
| paraphrase | One clear natural question about what the 10-K discloses; expand abbreviations ("cap allocation" → "capital allocation"); add no facts |
| hyde | A 3-4 sentence passage in 10-K style that would answer the question (HyDE); no specific numbers or names |
| multi | 3 search queries, each aimed at a different 10-K section, in filing vocabulary; one per line |

Every prompt states the company and ticker ("The company is Eversource Energy
(ticker ES); ES refers to this company"); in production the entity resolver
supplies this. Without it the model expanded "ES" into "Environmental
Settlements". Ollama returns HTTP 501 for JSON-schema output with gemma4, so
the multi-query prompt asks for plain lines. Cost: ~11 s per question for the
three calls on the M1 Pro.

**Scoring.** Dense Qwen (instr1) inside the ticker filter, then MMR (λ 0.85,
pool 60), as production. With several query vectors, a chunk's relevance is
its max cosine over them (or the mean, for `all_mean`). Baseline `orig` =
production. HyDE is embedded both as a query (`hyde_q`, with the instruction)
and as a document (`hyde_d`, no instruction).

**Dev (122 scorable; 34 "deep": gold ranked >60 for the original question).**
Variants with `_mean` average the cosines over their query set; the others take
the max. `hyde_q` = HyDE embedded as a query, `hyde_d` = as a document.

| Variant | Hit@8 | MRR@8 | Δ vs orig [95% CI] | W/L | Hit@8 on deep | LLM calls |
|---|---|---|---|---|---|---|
| orig (production: original question + MMR) | 0.590 | 0.354 | — | — | 0.000 | 0 |
| **orig+multi+hyde_q_mean** | **0.811** | 0.590 | **+0.221 [+0.139, +0.303]** | 29/2 | 0.529 | 2 |
| orig+multi+hyde_q (max) | 0.795 | 0.534 | +0.205 [+0.139, +0.279] | 25/0 | 0.412 | 2 |
| all_mean (orig, para, hyde_d, multi) | 0.795 | 0.594 | +0.205 [+0.131, +0.287] | 27/2 | 0.471 | 3 |
| **orig+hyde_q_mean** | 0.787 | 0.549 | +0.197 [+0.123, +0.270] | 25/1 | 0.353 | 1 |
| all_q_mean (orig, para, hyde_q, multi) | 0.787 | 0.568 | +0.197 [+0.123, +0.279] | 26/2 | 0.441 | 3 |
| **orig+multi_mean** | 0.787 | 0.593 | +0.197 [+0.115, +0.287] | 27/3 | 0.471 | 1 |
| hyde_q alone | 0.779 | 0.541 | +0.189 [+0.098, +0.279] | 30/7 | 0.559 | 1 |
| orig+hyde (max, hyde_d) | 0.754 | 0.473 | +0.164 [+0.098, +0.238] | 22/2 | 0.353 | 1 |
| orig+multi (max) | 0.754 | 0.484 | +0.164 [+0.107, +0.238] | 20/0 | 0.353 | 1 |
| all (max) | 0.746 | 0.451 | +0.156 [+0.074, +0.238] | 23/4 | 0.382 | 3 |
| orig+hyde_q (max) | 0.738 | 0.525 | +0.148 [+0.082, +0.213] | 19/1 | 0.294 | 1 |
| all_q (max) | 0.738 | 0.458 | +0.148 [+0.074, +0.221] | 21/3 | 0.324 | 3 |
| hyde_d alone | 0.705 | 0.473 | +0.115 [+0.016, +0.213] | 27/13 | 0.471 | 1 |
| para alone | 0.590 | 0.353 | 0.000 [−0.074, +0.074] | 12/12 | 0.235 | 1 |
| orig+para (max) | 0.574 | 0.339 | −0.016 [−0.082, +0.049] | 8/10 | 0.147 | 1 |

Dev findings:
- Rewriting is the largest single retrieval gain so far (+0.15 to +0.22 Hit@8,
  on top of MMR), and it reaches the deep misses (up to half of them).
- HyDE and multi-query both work; paraphrasing alone does nothing. The gain
  comes from writing the query in *filing vocabulary*, not from fixing grammar.
- HyDE works better embedded as a query (with the instruction) than as a document.
- Averaging similarity across the query set beats taking the max: max lets
  one off-target rewrite pull in its own neighbours.
- 15 variants were compared on dev, so the best dev number is optimistic.

**Pre-registered for test** (chosen before running test): orig+multi+hyde_q_mean
(best, 2 calls), orig+hyde_q_mean (1 call), orig+multi_mean (1 call).

**Test (126 scorable; 37 deep)** — run once, after pre-registration.

| Variant | Hit@8 | MRR@8 | Δ vs orig [95% CI] | W/L | Hit@8 on deep |
|---|---|---|---|---|---|
| orig (production) | 0.595 | 0.384 | — | — | 0.000 |
| **orig+hyde_q_mean** (pre-registered, 1 call) | **0.762** | 0.501 | **+0.167 [+0.079, +0.254]** | 29/8 | 0.459 |
| orig+multi+hyde_q_mean (pre-registered, 2 calls) | 0.746 | 0.488 | +0.151 [+0.056, +0.246] | 30/11 | 0.541 |
| orig+multi_mean (pre-registered, 1 call) | 0.722 | 0.477 | +0.127 [+0.040, +0.222] | 26/10 | 0.514 |
| *not pre-registered, for reference:* | | | | | |
| hyde_q alone | 0.770 | 0.507 | +0.175 [+0.071, +0.278] | 34/12 | 0.649 |
| orig+hyde_q (max) | 0.762 | 0.470 | +0.167 [+0.079, +0.246] | 27/6 | 0.486 |
| all_mean | 0.754 | 0.501 | +0.159 [+0.071, +0.246] | 29/9 | 0.541 |
| hyde_d alone | 0.714 | 0.451 | +0.119 [+0.016, +0.222] | 30/15 | 0.568 |
| orig+para | 0.540 | 0.309 | −0.056 [−0.119, +0.008] | 6/13 | 0.135 |
| para alone | 0.532 | 0.306 | −0.063 [−0.143, +0.016] | 9/17 | 0.189 |

Test findings:
- All three pre-registered variants hold, with CIs well above zero. The dev
  winner (2 calls) shrank most (+0.221 → +0.151); the 1-call orig+hyde_q_mean
  was the most stable (+0.197 → +0.167) and is the cheapest.
- Multi-query adds nothing over HyDE on test.
- HyDE alone rescues the most deep misses but also loses the most questions
  (12 on test); keeping the original question in the mean protects them.
- `orig` here (0.595) differs slightly from §6's MMR test number (0.603)
  because §6 ran MMR over a deduplicated pool; this lab, like production,
  runs MMR over the raw top 60.

**Decision.** orig+hyde_q_mean: `retrieval.query_expansion: hyde`
(`retrieval/rewrite.py` `HydeExpander`; `HybridRetriever` scores chunks by the
mean cosine over {question, passage}, for the dense search and the MMR
relevance). The passage is only a retrieval key; it is never shown to the
generator. Expansion failure falls back to the question alone. Tests:
`tests/test_retrieval.py::test_expansion_pulls_in_vocabulary_match`,
`::test_hyde_expander_prompt_and_failure`.

**Production check** (dev, the real pipeline retriever with HyDE + MMR, generating its own
passages through the pipeline's LLM client): Hit@8 0.803 vs lab 0.787; agrees with the lab
on 120/122 questions. Latency: median 19 s per retrieval vs ~5 s for the lab's HyDE call,
not yet explained (candidates: memory pressure from the full chunk catalog next to the
model on 16 GB; the pipeline's 8k context vs the lab's 2k). For scale, a v2 end-to-end
question took ~95 s.

### 7.1 Robustness: memory or style? Do natural questions benefit?

Two worries about HyDE: (a) the gain may come from gemma recalling the named
company from pre-training rather than from 10-K *style*; (b) FinDER questions are
terse analyst shorthand, so HyDE might not help, or might hurt, ordinary questions.

**Company-blind HyDE.** The question is masked before generation (every mention of
the company: registered name, short alias, ticker, initials, suffixes →
"the company"; 0 leaks over 621 questions), and the prompt names no company.
Code: `EntityResolver.mask` (`retrieval/entity.py`), `tools/rewrite_lab.py generate --blind`.

**Natural questions.** `eval/synth_claude.json`: 361 questions written by Claude from
single passages, company-specific and often numeric ("How many acquisitions did Arthur
J. Gallagher (AJG) complete in 2024, and what was the total net consideration paid?").
Gold = the source passage. Easier than FinDER (orig Hit@8 0.825; only 22 deep misses).

All rows use MMR (λ 0.85, pool 60); Δ vs orig with 95% bootstrap CI; W/L = questions won/lost.

| Variant | FinDER all (248) Hit@8 | Δ | W/L | Synth (361) Hit@8 | Δ | W/L |
|---|---|---|---|---|---|---|
| orig | 0.593 | — | — | 0.825 | — | — |
| orig+hyde_q_mean (company named) | 0.774 | +0.181 [+0.125, +0.238] | 54/9 | **0.753** | **−0.072 [−0.116, −0.025]** | 26/52 |
| orig+hyde_blind_mean (company-blind) | 0.758 | +0.165 [+0.109, +0.222] | 49/8 | **0.972** | **+0.147 [+0.114, +0.186]** | **53/0** |
| hyde_q alone | 0.774 | +0.181 | 64/19 | 0.576 | −0.249 | 27/117 |
| hyde_blind alone | 0.750 | +0.157 | 59/20 | 0.975 | +0.150 | 58/4 |

Findings:
- **(a) It is style, not memory.** On FinDER, removing the company keeps 91% of the
  gain (+0.165 vs +0.181; CIs overlap almost entirely).
- **(b) Naming the company breaks HyDE on natural questions.** gemma4:12b answered
  168 of 361 synth prompts with a refusal ("Please provide the relevant text from the
  CME Group 10-K filing. I cannot answer this question because no text was
  provided..."). The prompt's "never answer from memory" plus a named company and a
  specific factual question reads as a request for facts it does not have. FinDER's
  telegraphic questions rarely trigger this (1/260). Split by refusal (synth):

  | Synth subset | n | orig | orig+hyde_q_mean | orig+hyde_blind_mean |
  |---|---|---|---|---|
  | named-company passage OK | 193 | 0.860 | 0.969 | 0.979 |
  | named-company passage = refusal | 168 | 0.786 | 0.506 | 0.964 |

  Where gemma did write a passage, both versions help about equally. The whole loss
  comes from averaging a refusal into the query vector. Company-blind prompts refused
  on 1/361 synth and 0/260 FinDER.
- On synth, company-blind HyDE never loses a question (53/0); it lifts Hit@8 to 0.97.

**Why naming the company hurts (mechanism).** The name itself does not hurt retrieval;
it makes the generator refuse, and the refusal text is what hurts.
1. *Trigger.* The system prompt says "You never answer from memory and never add facts
   that are not in the user's text" (there to stop invented facts). Combined with a
   specific factual question about a real, named company, the model reads the task as
   "state CME's actual buyback, without memory" and asks for the document instead.
   With "the company" there is no real entity to be wrong about, so the task reads as
   "write a generic 10-K-style paragraph", which it does. Same question (synth_s0002):

   | Prompt question | gemma4:12b output |
   |---|---|
   | "What share repurchase program did **CME Group's** board approve in December 2024, and how large is it?" | "Please provide the relevant text from the CME Group 10-K filing. I cannot answer this question because no text was provided for me to reference." |
   | "What share repurchase program did **the company's** board approve in December 2024, and how large is it?" | "In December 2024, the Board of Directors authorized a share repurchase program to acquire the company's common stock... up to a certain amount of its outstanding shares..." |

2. *Why it depends on the question style.* FinDER questions are telegraphic and vague
   ("Cap allocation impact on investor confidence, MCD"): nothing specific to get wrong,
   1/260 refusals. Synth questions ask for exact facts ("how many acquisitions... total
   consideration?"): 168/361 refusals. An evaluation on FinDER alone would not have
   shown the failure.
3. *Why a refusal is worse than no expansion.* Dense search uses the mean of the
   question and passage vectors. A refusal passage pulls that mean halfway toward
   chunks that resemble "please provide the text / cannot answer", pushing the gold
   chunk out of the top 8 (refused subset: 0.786 → 0.506).

Lessons: (i) an anti-hallucination instruction can make a rewriter refuse instead of
rewrite; keep entity names out of a rewriting prompt when a filter already scopes the
search. (ii) A query expansion must fail closed: an expansion that returns junk
should be dropped, never averaged in. (iii) Validate a rewriting step on more than one
question style; the benchmark's own style can hide the failure mode.
- Generation time per HyDE call: 5–9 s (gemma4:12b, num_ctx 2048).

**Decision (revised).** Production HyDE is company-blind: `HydeExpander(llm,
mask=resolver.mask)` masks the question for the retrieval filter's ticker(s) and any
company the resolver finds; the prompt no longer names the company (the old `WHO` line
was removed). The ticker filter already scopes the search, so the passage only has to
supply vocabulary. Masking also makes the "ES" → "Environmental Settlements" failure
impossible (no ticker reaches the model). Refusal guard: `rewrite.is_refusal`
discards a refusal-like passage and retrieval falls back to the question alone
(catches all 168 named-company refusals and the single blind refusal; no false positives
on the remaining 620 blind passages). Expected cost on FinDER: −0.016 Hit@8 vs the
named version, inside noise. Tests: `test_hyde_expander_prompt_and_failure`
(asserts the prompt is company-blind), `test_hyde_expander_drops_refusals`.

**Status:** company-blind version implemented and tested (67 tests pass). Still off by
default: the 19 s production latency is unexplained, and the production check above
used the named-company prompt, so it needs re-running.

### 7.2 Company-blind HyDE in production; latency; v3 and v3_gold

**Production check** (real pipeline retriever, `HydeExpander` with `resolver.mask`,
MMR; `out/rewrite/prod_check_blind.log`):

| Set | orig (lab) | Production blind HyDE | Agrees with lab | Empty expansions |
|---|---|---|---|---|
| FinDER dev (122) | 0.590 | 0.787 | 118/122 | 0 |
| Synth, first 80 | 0.825 | 0.975 | 80/80 | 0 |

**Latency explained.** Median retrieval 19.0 s (FinDER) / 17.1 s (synth), of which the
gemma call is 7.6 / 8.5 s. Without HyDE the same retriever takes 0.2 s per question.
Profile of 7 HyDE retrievals: Ollama HTTP wait 4.3 s/question, Chroma query 3.7 s
(0.16 s without HyDE), torch `.cpu()` sync 1.6 s (0.05 s), with 5.6 GB of swap in use.
The non-LLM overhead is memory contention: gemma4:12b, the Chroma HNSW index and the
fp32 query embedder do not fit together in 16 GB, so each component pages the others
out. Not a code problem; options if it matters: fp16 query embedder (−1.2 GB), a 2k
context for the HyDE call (pipeline uses 8k), or HyDE on a smaller model. For scale,
a full v2 answer took ~95 s.

**Decision:** company-blind HyDE enabled (`query_expansion: hyde`) in
`config/local_16gb.yaml`, `local_16gb_v3.yaml`, `local_16gb_v3_gold.yaml`.

**Re-measured on v3 and v3_gold** (all FinDER questions; MMR in every row except "base";
`out/diversity/sec_index_v3*_all.log`, `out/rewrite/eval_sec_index_v3*_*.log`):

| Hit@8 | qwen06 (v2 corpus) | v3 | v3_gold (gold passage guaranteed in index) |
|---|---|---|---|
| Scorable FinDER questions | 248 | 249 | 260 |
| base (no MMR) | ≈0.524 (dev .525 / test .524) | 0.510 | 0.535 |
| MMR λ 0.85 (diversity lab) | ≈0.597 (dev .590 / test .603) | 0.598 (+0.088 [+0.052, +0.129], 24/2) | 0.627 (+0.092, 26/2) |
| orig, MMR (rewrite lab) | 0.593 | 0.586 | 0.608 |
| orig+hyde_blind_mean | 0.758 | 0.775 (+0.189 [+0.137, +0.249], 54/7) | 0.804 (+0.196, 58/7) |
| orig+hyde_q_mean (named) | 0.774 | 0.775 | 0.796 |
| Synth (361): orig → orig+hyde_blind_mean | 0.825 → 0.972 | 0.831 → 0.964 (48/0) | same as v3 (no synth gold added) |

Findings:
- Both fixes replicate on the new corpus with the same size of gain; on v3 the blind and
  named HyDE variants tie on FinDER (0.775), and blind still wins decisively on synth.
- **Coverage is not the bottleneck; ranking is.** With the exact gold passage
  guaranteed to be in the index (v3_gold), plain dense retrieval finds it in the top 8
  for only 53.5% of questions — barely above v3 (51.0%). Even with MMR + HyDE, 20% of
  questions miss a passage that is certainly there. The remaining retrieval error is
  the embedder ranking other chunks of the same company above the gold one (median
  gold rank 7–8 without HyDE, 2–3 with it).
- v3's base is slightly lower than qwen06's (0.510 vs ≈0.524): more filings per ticker
  (the 2022 10-Ks) means more near-duplicate competitors for the same top 8. MMR
  absorbs this (0.598 vs ≈0.597).

## 8. Answer stage: verifier, citations, abstention (Nov 11)

Fixes for §4.3–4.4, implemented in code; measured so far only offline (re-scoring the
stored v2 judge output), because an end-to-end run has to wait for the v3 index.

**8.1 Quote matching** (`generation/verifier.py` `quote_matches`). A judge's quote counts
if it occurs in the chunk after whitespace normalization (as before), or, for quotes of
≥ 5 words, if its word sequence occurs contiguously ignoring punctuation, case, curly
quotes and dashes. Reordered, paraphrased or absent words still fail.

**8.2 Citation re-pointing** (`CitationVerifier._repoint_citation`). If a SUPPORTED/
PARTIAL claim's quote is not in the cited excerpt (or no excerpt is cited) but is in
another retrieved excerpt, the citation is moved there and `citation_corrected` is set;
the numeric check then runs against the corrected excerpt. Reports carry
`citations_corrected`.

**8.3 Meta-claims** (`is_meta_claim`). "The provided excerpts do not contain information
regarding…" describes the evidence, not the company. Such claims are flagged `meta` and
excluded from total/supported/groundedness; a sentence with "rather"/"; they…" (a
positive claim folded in) is kept. An answer whose claims are *all* meta abstains
("answer only describes missing evidence").

**8.4 Grouped citation markers (bug found while doing 8.2).** Citations were parsed with
`\[(\d+)\]`, so `[1, 2]`, `[1,2]` and `[2; 4]` were never read. In v2, 24 of 266
generations cited *only* in grouped form (their answer citations silently fell back to
excerpts 1–2) and 76 more mixed both forms (some citations dropped). Now
`verifier.cited_numbers` parses single, grouped and range markers; `grounded.py` uses it.
This affects the citation list and the `min_citations` gate, not the judge (which reads
the raw text).

**8.5 Partial answers instead of all-or-nothing abstention** (`grounded.strip_insufficient`).
`INSUFFICIENT_EVIDENCE` now abstains only when the remaining text has no citation or
fewer than 8 words; otherwise the flag is removed, `partial_insufficient` is recorded,
and the cited remainder goes through the verifier and the groundedness gate as usual
(also applied to the repair turn). Replayed on the v2 generations: all 6 placebos and
all 125 bare-flag answers still abstain; 7 answers become partial answers (finder_0007,
0042, 0058, 0079, 0085, 0089, 0142).

**8.6 Generator prompt** (`prompts.SYSTEM_PROMPT`). Old: "If the excerpts do not contain
enough information to answer, reply exactly: INSUFFICIENT_EVIDENCE." New: answer the
covered part with citations and say in one sentence what the excerpts do not cover
(no inference to fill the gap); reply exactly `INSUFFICIENT_EVIDENCE` if nothing in the
excerpts answers any part, *or if the question rests on a premise the excerpts do not
support (a period, document, event or company not in them)*. The premise clause is
there for the placebos, which are all false-premise traps (Apple's 1999 10-K, Tesla's
2024 bankruptcy, NVIDIA's fiscal-2028 forecast...). Target: the 40 bare class-B
abstentions ("investor confidence", "stock performance") where the filing states the
underlying facts. Risk: placebos answered with related facts. **Untested end to end;
the placebos must stay 6/6.**

**Offline re-score of v2** (stored judge verdicts replayed through the new verifier;
a claim stored UNSUPPORTED with a quote that failed the old check is assumed judged
SUPPORTED, so this is an upper bound; `tools/rescore_verifier.py`):

| | Old rules | New rules |
|---|---|---|
| Claims scored | 811 | 788 (23 meta excluded) |
| Citations re-pointed | — | 10 |
| Supported share | 0.845 | 0.928 |
| Answers passing the gate (no contradiction, groundedness ≥ 0.6), of 123 | 113 | 119 |

The 6 answers that now pass: finder_0004, 0100, 0106, 0179, 0183, 0186.

Tests: `tests/test_verification.py::test_quote_match_ignores_punctuation_case_and_typography`,
`::test_citation_repointed_to_chunk_holding_quote`, `::test_meta_claims_excluded_from_groundedness`,
`::test_grouped_citation_markers_parsed`, `::test_insufficient_flag_only_abstains_when_bare`
(72 tests pass).

## 9. End-to-end v3: all fixes (Nov 12)

**Run.** `out/provenance_16gb_v3`, `config/local_16gb_v3.yaml`: v3 corpus (§5.4),
MMR (§6), company-blind HyDE (§7.2), the answer-stage changes of §8 (prompt,
verifier, abstention rules), fp16 query embedder. 266 questions (260 FinDER + 6
placebos), Nov 11 21:57 → Nov 12 07:16, ≈2.1 min per question, median answer
latency 21 s. The gold-passages run (`out/provenance_16gb_v3_gold`) follows.

**Memory first.** The first launch was killed by Claude Code's low-memory reaper,
and the relaunch thrashed: the Python process alone used 10.0 GB beside the
9.3 GB qwen3:14b judge on a 16 GB machine (swap 9.9 of 10.2 GB, one question in
~20 min). Profile by stage (`footprint`):

| Stage | Before | After |
|---|---|---|
| Query embedder (fp32 → fp16) | 3.7 GB | 2.1 GB |
| Chunk catalog (729,775 chunks) | 2.9 GB | 0.36 GB |
| Chroma HNSW index (first query) | 3.3 GB | 2.9 GB |
| **Process total** | **10.0 GB** | **5.3 GB** |

- The catalog was mostly chunk text: 71% of chunks contain a curly quote or dash,
  so CPython stores them at 2 bytes/char (2.0 GB). `SECIndex.load` now keeps
  metadata-only stubs (one shared frozen `FilingMetadata` per filing) and `get()`
  reads a chunk's text from `chunks.jsonl` by offset; `catalog()` serves the
  metadata-only callers (entity resolver, ticker filter). 2,000 random chunks match
  the file exactly; `tests/test_sidecar.py::test_loaded_index_reads_text_lazily_and_round_trips`.
- fp16 queries vs fp32 on the v3 index (whose documents were embedded in fp16):
  cosine 1.00000, identical top-8 on 255/260 FinDER questions. The 28 questions
  done with fp32 were set aside (`out/provenance_16gb_v3_fp32_partial`) so the
  run uses one setting throughout.
- After the fix: pressure level 1–2 ("normal"/"warn") all night, one transient
  critical reading when Ollama swapped judge → generator, swap 3–5 GB.

**Report fix: meta claims.** The v3 prompt asks for one sentence on what the
excerpts do not cover. The verifier excludes such meta claims from groundedness
(§8.3), but `tools/provenance_report.py` still counted them as unsupported, and
the meta rule missed gemma4's main phrasing, "The excerpts do not **cover** …"
(95 of 261 unflagged claims whose subject is the excerpts). As first reported, v3
looked like a collapse (faithfulness 0.706, 6 clean answers, placebos 5/6). Fixes:
`is_meta_claim` matches "excerpts/documents … do not [adverb] <any verb>" and
inventory sentences ("the documents include reports for fiscal years 2021, 2022
and 2024 …", which made placebo_0001 count as answered); the report re-applies the
current rule to every run (`tests/test_verification.py::test_meta_claims_excluded_from_groundedness`).

**Results** (both runs scored by the same report and meta rule):

| Metric | v2 | v3 |
|---|---|---|
| Answered | 115 | **213** |
| Faithfulness (all claims, incl. answers the gate then rejected) | 0.863 | 0.862 |
| Context recall, strict / cover | 0.235 / 0.473 | **0.373 / 0.706** |
| Weighted claim error rate | 0.274 | 0.277 |
| Hardened-qualifier claims | 39 | 73 |
| Placebos refused | 6/6 | 5/6 as run; 6/6 with the current meta rule |
| Wrong-period answers (year-named questions) | 1/3 | 4/6 |

Error sources:

| Source | v2 | v3 |
|---|---|---|
| clean | 57 | **111** |
| retrieval miss → abstained | 97 | **18** |
| retrieval miss → ungrounded claims | 18 | 20 |
| abstained (gate/judge) | 48 | 30 |
| generation: unsupported claims | 27 | 43 |
| generation: dropped qualifiers | 12 | 35 |
| wrong-period evidence | 1 | 3 |
| placebos: correctly abstained / answered | 6 / 0 | 5 / 1 (meta-only, see above) |

Paired, FinDER only: 107 questions answered by both, 105 abstained in v2 and
answered in v3, 8 answered in v2 and abstained in v3, 40 abstained in both. Claim
support (meta excluded) on the 107: v2 0.898 → v3 0.935; on the 105 new answers:
0.912. v3 abstentions (48 FinDER): 39 by the groundedness gate, 5 bare
INSUFFICIENT_EVIDENCE, 4 meta-only. Citations re-pointed to
the chunk holding the quote: 13.

**Reading.**
- Retrieval was the binding constraint: retrieval-miss abstentions fell 97 → 18
  and gold coverage 0.47 → 0.71 (MMR + HyDE + v3 corpus). Answer quality per claim
  held or improved while answers almost doubled.
- Dropped qualifiers grew faster than answers (39 → 73 claims; 12 → 35 questions
  where it is the main error). The qualifier-preservation repair is the next
  generation-side fix.
- Wrong period: the counter is crude (any cited filing > 1 year from a year in the
  question). finder_0093's "2055" is an asbestos-liability horizon; finder_0078
  and 0221 cite a 2025 filing beside the 2023 ones; placebo_0001 is the meta-only
  case. Period filters (open) would address the mixed-year citations.
- placebo_0001 is a correct refusal in substance ("the excerpts do not contain any
  information regarding revenue forecasts for fiscal year 2028"); it was counted as
  an answer only because the second sentence escaped the meta rule. The rule is now
  fixed in the pipeline, but this run was made with the old one.

## 10. End-to-end v3_gold: gold passages in the index (Nov 12)

Same pipeline as v3 (`config/local_16gb_v3_gold.yaml`), on the index with the
FinDER gold passages added as chunks (§7.2), 266 questions, finished Nov 12 16:51.
Scored with the same report (`out/provenance_16gb_v3_gold/report_cover.txt`).

**Report fixes found by this run.** placebo_0004 ("impairment charges after
Tesla's 2024 bankruptcy filing") answered "The provided excerpts do not contain
any information regarding a 2024 bankruptcy filing by Tesla, nor do they describe
any impairment charges …". The judge split it into two claims and the second,
"nor do they describe …", escaped the meta rule: `is_meta_claim` now also matches
"nor do/does they/it/the excerpts <verb>" (test added). The report also counted
a meta-only answer to a placebo as CONFABULATED; it now counts as a refusal
(in v3 this was placebo_0001; both runs are re-scored below).

| Metric | v3 | v3_gold |
|---|---|---|
| Answered | 213 | 215 |
| Faithfulness (all claims) | 0.863 | 0.849 |
| Context recall, strict / cover | 0.373 / 0.706 | 0.408 / **0.754** |
| Weighted claim error rate | 0.274 | 0.301 |
| Hardened qualifiers (lexical) | 73 | 68 |
| Placebos refused (incl. meta-only answers) | 6/6 | 6/6 |
| Wrong-period answers | 4/6 | 3/7 |

| Error source | v3 | v3_gold |
|---|---|---|
| clean | 112 | **126** |
| retrieval miss → abstained | 18 | 17 |
| retrieval miss → ungrounded claims | 20 | 14 |
| abstained (gate/judge) | 30 | 30 |
| generation: unsupported claims | 41 | 38 |
| generation: dropped qualifiers | 36 | 33 |
| wrong-period evidence | 3 | 2 |

(v3's numbers differ by one or two from §9 because of the widened meta rule.)

Paired, FinDER (260): gold coverage of the evidence rose for 15 questions and fell
for 2; questions with cover ≥ 0.5: 186 → 199. Claim support on answered questions
(supported + ½ partial, meta excluded): 0.927 → 0.933. 197 answered in both, 16
newly answered, 15 lost.

**Run-to-run noise.** 80 FinDER questions got the identical evidence list in both
runs. Among them only 41 got the identical answer text, 9 changed error source
and 3 flipped abstention: generation is not deterministic at temperature 0 on
this stack (cause not isolated; Ollama on Metal). So roughly ±10% of per-question labels change with no input
change; differences of a few questions between end-to-end runs (e.g. "lost 15 /
new 16" above) are within noise. Paired, retrieval-level labs (deterministic) are
the reliable instrument; end-to-end runs show only large effects.

**Reading.** Guaranteeing that the gold passage exists lifts coverage by 0.05 and
clean answers by 14, mostly through the 15 questions whose evidence gained the
gold passage. The remaining gap is ranking (§7.2: 20% of FinDER questions still
miss the gold passage in the top 8 with it in the index) and generation
(unsupported claims and dropped qualifiers, ~70 questions), not corpus coverage.

## 11. Dropped qualifiers: detection and repair (Nov 12)

A claim "drops a qualifier" when it states as fact what the filing hedges
("competition may further intensify" → "competition is intensified"; "not
expected to be material" → "no material impact"). It is supported word for word
by the judge but says something the filing does not.

**How many are real.** The report's lexical proxy (the supporting quote has a hedge
word, the claim has none) flagged 73 claims in v3. Hand labels for all 73
(`eval/qualifier_labels.json`, written in session): **14 real (precision 0.19)**.
Real examples: finder_0025 "no material impact in fiscal 2024" from "not expected
to be material"; "Success depends on …" from "We believe that … depends upon";
"weak economic growth has intensified …" from "may further intensify". False
alarms: hedges in another clause ("liabilities that *may* arise"), adjectives
("*potential* vulnerabilities"), "*subject to* competition", attributions ("the
company states …"). A clause-anchored lexical rule reached only precision 0.50 /
recall 0.50, because real cases are paraphrased; abandoned.

**Design** (`generation/qualifiers.py`, opt-in, `config/local_16gb_v4.yaml`):
1. Prevention: a system-prompt rule to keep the filing's qualifiers
   (`GenerationConfig.qualifier_rule`).
2. Detection: the lexical proxy picks candidates; one short judge call per answer
   (qwen3:14b, separate prompt so the main verdicts stay comparable) labels each
   DROPPED / OK (`VerificationConfig.qualifier_check`).
3. Repair: one re-prompt naming the dropped hedges; kept only if groundedness does
   not fall, contradictions do not rise and fewer claims drop qualifiers
   (`qualifier_repair`). Each answer records `extra["qualifiers"]`.

**Judge check vs hand labels** (`tools/qualifier_lab.py`, `out/qualifier_lab.txt`):

| Detector | Precision | Recall |
|---|---|---|
| Lexical proxy | 0.19 | 1.00 (by construction) |
| qwen3:14b check, prompt v1 (DROPPED/OK with examples) | 0.67 (2/3) | 0.14 (2/14) |
| qwen3:14b check, prompt v2 (locate the governing hedge, then decide) | 0.55 (6/11) | **0.43 (6/14)** |

Prompt v1 is precise but almost never says DROPPED: it misses "can harm" from
"could harm", "Success depends on" from "We believe … depends upon", and "did not
have a material impact" from a hedged expectation, cases the prompt names as
examples. Prompt v2 (`QUALIFIER_PROMPT_V2`: (a) find the quote words behind the
claim, (b) is a hedge governing them, (c) does the claim keep or attribute it;
reply `n: hedge=…; kept=…; DROPPED|OK`) triples recall. Its false alarms are
mostly "is involved in legal proceedings" claims from "proceedings that may
arise". Caveats: v2 was written after reading v1's misses on the same 73 labels
and there are only 14 positives, so 0.43 is optimistic. A false alarm costs one
re-prompt that is kept only if it does not lower groundedness, so v2 is the default
from now on (`QualifierChecker(prompt="v2")`). The v4 subset run below was started
with v1, so it mainly measures the prevention rule.

**v4 on the 56 flagged questions** (`out/provenance_16gb_v4_qualsub`, prompt rule +
check v1 + repair, Nov 12 17:00–19:37). The 56 were chosen because v3 had lexical
flags there, so a plain rerun would already show fewer (regression to the mean).
The control is v3_gold on the same 56 questions (same prompts, no qualifier fix):

| Same 56 questions | v3 (selection run) | v3_gold (control) | v4 subset |
|---|---|---|---|
| Answered | 53 | 49 | 49 |
| Claims in answered | 300 | 295 | 350 |
| Claim support (supported + ½ partial) | 0.933 | 0.920 | 0.924 |
| Lexical hardened claims / answers with any | 73 / 56 | 49 / 33 | **34 / 24** |
| Main error "dropped qualifiers" | 36 | 22 | **12** |
| Clean | 0 | 11 | **15** |
| Judge-checked drops (check v1) / repaired | – | – | 2 / 1 |

Of the 11 questions with a hand-labelled real drop in v3, 8 have no lexical
candidate left in v4 and 3 are clean. Versus the control the prompt rule cuts
lexical flags by about 30% and "dropped qualifiers" as main error from 22 to 12,
without lowering support. Answers got longer (350 vs 295 claims), so a few more
unsupported claims (26 vs 23). The check v1 found 2 drops, and the one repair was
kept; with v1's recall (0.14) this run tells little about repair, so the
overnight run uses check v2.

## 12. Period filters (Nov 12)

**Problem.** Year-named questions ("in fiscal 2023") retrieve from every year of the
ticker's filings; v3 had 4 of 6 such FinDER answers citing a filing more than a year
off (§9, crude counter).

**Implementation** (`retrieval/period.py`, `RetrievalConfig.period_filter`,
`period_mode`, `period_boost`). `named_years` reads 4-digit years (1990–2039, not
amounts like "$2023 million" or "2,024") and FY24 / fiscal '24. A question about Y
is answered by the filings of fiscal years Y and Y+1 (the next 10-K restates Y as
the comparative). Only years the ticker has are kept; if none exists ("fiscal
2028") nothing is filtered, so the generator sees the nearest filings and can say
so. Hard mode adds a `fiscal_year $in` filter. Soft mode keeps every year and adds
`period_boost` to the cosine of in-period chunks, in both the 60-candidate pool
(which is the union of the ticker-filtered and the period-filtered top 60) and the
MMR relevance. The applied period is recorded as `extra["period_filter"]`.

**Lab** (`tools/period_lab.py`, production pool: mean cos(question, company-blind
HyDE), MMR λ 0.85; `out/period/`):

| 228 year-named synth questions | Hit@8 | MRR@8 | In-period share of top 8 | ΔHit [95% CI] | W/L |
|---|---|---|---|---|---|
| prod (no period filter) | 0.952 | 0.837 | 0.69 | – | – |
| hard, fiscal_year = Y | 0.965 | 0.850 | 0.975 | +0.013 [−0.009, +0.035] | 5/2 |
| hard, {Y, Y+1} | 0.961 | 0.839 | 0.987 | +0.009 [−0.009, +0.026] | 3/1 |
| soft +0.02 | 0.961 | 0.838 | 0.80 | +0.009 | 3/1 |
| soft +0.05 | 0.961 | 0.844 | 0.89 | +0.009 | 3/1 |
| soft +0.1 | 0.961 | 0.842 | 0.954 | +0.009 [−0.009, +0.026] | 3/1 |

The source chunk of the synth questions is in a filing of the named year for 195
of 227, Y+1 for 12, Y−1 for 11. FinDER has only 4 scorable year-named questions;
soft at every boost keeps all 4 hits, while both hard filters lose one (its gold
passage sits in another year's filing).

**Reading.** Hit@8 on synth year-named questions is already 0.95 (written from one
chunk, the questions are easy), so the gain there is within noise. What the filter
changes is where the rest of the top 8 comes from: 69% → 95% from the named period.
That is the evidence mix behind the wrong-period answers. Soft +0.1 gets nearly all
of the hard filter's period purity without dropping off-period gold, so it is on in
the v4 config. Whether it lowers wrong-period answers end to end will show in the
v4 run, but with 6–7 year-named FinDER questions it can only be anecdotal.

## 13. Cross-encoder reranking (Nov 12)

**Question.** With the gold passage in the index (v3_gold) about 20% of FinDER
questions still miss it in the top 8 (§7.2), a ranking problem. Does a
cross-encoder over the production pool fix it?

**Lab** (`tools/rerank_lab.py`, `out/rerank/`): the production pool (top 60 by mean
cos(question, company-blind HyDE) inside the ticker filter) re-scored with three
cross-encoders: ms-marco-MiniLM-L-6-v2, bge-reranker-v2-m3 and Qwen3-Reranker-0.6B.
Variants: top 8 by reranker score (`rr_`), MMR with the min-max scaled reranker
score as relevance (`rr_*_mmr`), and reciprocal-rank fusion of dense and reranker
ranks (`rrf_`). Paired against production (MMR λ 0.85 on the dense score). v3
index, 249 scorable FinDER questions (gold in the pool for 222):

| Variant | Hit@8 | MRR@8 | ΔHit [95% CI] | W/L |
|---|---|---|---|---|
| **prod (dense + HyDE + MMR)** | **0.775** | **0.517** | – | – |
| rrf_qwen | 0.675 | 0.473 | −0.100 [−0.149, −0.052] | 7/32 |
| rrf_minilm | 0.671 | 0.449 | −0.104 [−0.149, −0.060] | 4/30 |
| rr_qwen_mmr | 0.655 | 0.437 | −0.120 [−0.177, −0.064] | 13/43 |
| rr_qwen | 0.635 | 0.413 | −0.141 [−0.201, −0.084] | 13/48 |
| rrf_bge | 0.635 | 0.465 | −0.141 [−0.193, −0.092] | 6/41 |
| rr_minilm_mmr | 0.562 | 0.367 | −0.213 | 10/63 |
| rr_minilm | 0.542 | 0.358 | −0.233 | 10/68 |
| rr_bge_mmr | 0.534 | 0.353 | −0.241 | 10/70 |
| rr_bge | 0.518 | 0.347 | −0.257 | 10/74 |

**Reading.** Every reranker hurts, all CIs below zero. Even fusing the reranker
rank with the dense rank costs 0.10. A likely reason (not tested): the rerankers
score the terse FinDER question against the chunk alone, while the dense score
also uses the HyDE passage, which carries the 10-K vocabulary the question lacks
(§7). An earlier dev-split check (MiniLM +0.04) used a weaker baseline without
HyDE. The synthetic-set run was stopped: with this FinDER result it could not
change the decision, and it would have competed with the overnight run for
memory. No reranker in v4. (Untried: reranking with the HyDE passage appended to the
query.)

## 14. End-to-end v4: qualifier rule, check, repair, soft period filter (Nov 13)

`config/local_16gb_v4.yaml` = v3 + qualifier prompt rule + judge check (prompt v2)
+ one repair + soft period filter (+0.1), v3 index, 266 questions, Nov 12 21:10 →
Nov 13 07:04 (~2.3 min/question, median answer latency 21.8 s vs 21.4 s in v3).
Report: `out/provenance_16gb_v4/report_cover.txt`.

| FinDER (260) | v3 | v3_gold | v4 |
|---|---|---|---|
| Answered | 212 | 213 | 213 |
| Claim support (supported + ½ partial, meta excluded) | 0.927 | 0.933 | 0.925 |
| Claims per answer | 4.88 | 4.95 | **5.73** |
| Unsupported claims (per claim) | 76 (7.3%) | 70 (6.6%) | 91 (7.5%) |
| Lexical qualifier flags | 73 | 68 | **47** |
| **Real dropped qualifiers (hand-labelled)** | **14** | – | **4** (+2 borderline) |
| Judge-checked drops, check v2 on lexical candidates | 11 | 9 | 2 (circular: v4 repaired against this check) |
| Main error "dropped qualifiers" | 36 | 33 | 24 |
| Main error "unsupported claims" | 41 | 38 | 52 |
| Clean | 112 | 126 | 118 |

All 266 questions: answered 215, faithfulness 0.879 (v3 0.863), WCER 0.243 (v3
0.274). Repair was attempted and kept in 3 answers. Paired with v3: 188 answered in
both, 25 newly answered, 24 lost, in line with the noise in §10.

**Hand labels** (`eval/qualifier_labels_v4.json`, same rule as for v3): of v4's 47
lexical flags, 4 are real: "We believe the centralized scale … provides" →
"provides"; "intends to guide investments" → "uses … to guide"; "may include
briefings" → "through briefings"; "estimates that may take into account …" →
"based on estimates of …". Two borderline ("believe in the 70-20-10 approach",
"we believe we are better able to market") are not counted. The recall of the
lexical proxy itself is unknown (it only sees claims whose quote has a hedge word
and whose claim has none), so both 14 and 4 are lower bounds on the same
instrument.

**Placebos: 4/6 by the counter, 6/6 in substance.** placebo_0004 ("impairment
charges after Tesla's 2024 bankruptcy") says the excerpts have no bankruptcy
filing, then adds a true, supported fact (a $204M digital-asset impairment in
2022). placebo_0005 (Microsoft proxy CEO pay) says the figures are not in the
excerpts; one of its sentences ("the actual figures are not included in the
text") escapes the meta rule. Neither asserts the false premise. The counter
(abstained or meta-only) is too strict for refusals that add context; a
premise-check label would be the right measure.

**Period filter.** It applied to 7 of the 9 year-named questions (e.g.
finder_0036 "post-2025" → fiscal 2025) and, as designed, not to placebo_0001
(fiscal 2028) or placebo_0002 (1999), whose years the index does not have. Wrong-period counter 3/6 vs 4/6 in v3: finder_0221 now
abstains, finder_0036 is newly flagged (a 2023 filing among its citations; the boost
is soft). With so few questions this is anecdotal.

**Reading.**
- The qualifier fix works where it was aimed: real dropped qualifiers 14 → 4 on the
  same 260 questions, lexical flags −36%, with claim support unchanged. Most of it
  is the prompt rule; repair fired only 3 times.
- Side effect: answers got 17% longer (4.88 → 5.73 claims). Unsupported claims grew
  with them at the same rate (7.3% → 7.5% per claim), which moves questions into
  "unsupported claims" as the main error (41 → 52). The rule seems to make gemma
  quote more of the hedged context. A length cap or "one sentence per point" in
  the prompt would be the next thing to try.
- Clean answers 112 → 118 and the other end-to-end deltas are within the ±10%
  label noise (§10).

## 15. Not yet done / open

| Item | Status |
|---|---|
| Query rewriting | Done: company-blind HyDE validated (§7.1), production-checked and enabled in the 16 GB configs; latency explained (§7.2) |
| v3 corpus (10-K/A replaced, Exhibit 13, 2022 10-Ks) + v3_gold (FinDER passages added) | Refetch done (489 new 10-Ks, 463 of them filed in 2022). Embedding is compute-bound: Qwen3-Embedding-0.6B on the M1 Pro GPU runs ~3 chunks/s in bf16, ~4.6/s in fp16 (≈ 335 tokens/chunk), so all ≈163k new chunks would take ~10–15 h; the first attempt also thrashed (13 GB resident, swapping). Restarted with fp16 (cos to the stored bf16 vectors ≥ 0.9998) and only the 205 new filings of FinDER/synth tickers (~4 h); retrieval is ticker-filtered, so the other 284 filings cannot change any metric and are added later |
| Diversity / rewriting re-measured on v3 and v3_gold | Done (§7.2) |
| Verifier normalization, citation re-pointing, meta-claim exclusion, grouped citations | Implemented (§8); validated end to end (§9); meta rule widened Nov 12 |
| Abstention: prompt + abstain only on a bare flag | Validated end to end (§9): retrieval-miss abstentions 97 → 18; placebos 6/6 with the widened meta rule (5/6 as run) |
| Qualifier-preservation repair | Implemented (§11): prompt rule cuts lexical flags ~30% vs control on 56 questions; judge check v2 recall 0.43 on hand labels |
| Period filters | Implemented (§12): soft +0.1 boost on in v4 |
| Cross-encoder reranking | Tested (§13): every variant hurts (−0.10 to −0.26 Hit@8); not used. Untried: rerank with the HyDE passage in the query |
| Run-to-run noise | Measured (§10): ~11% of per-question labels change with identical evidence; use paired retrieval labs for small effects |
| 10-Q section segmentation | Lower priority now that diversification is label-free |
| End-to-end run with all fixes, v3 and v3_gold | Done (§9, §10) |
| End-to-end v4 (v3 + qualifier rule/check v2/repair + soft period filter) | Done (§14): real dropped qualifiers 14 → 4; answers 17% longer |
| Answer length after the qualifier rule | Open: try a length cap / one sentence per point (§14) |
| Placebo metric | Open: label refusals by whether the false premise is asserted, not by abstained/meta-only (§14) |
| Remaining 281 v3 10-Ks (non-eval tickers) | After the end-to-end runs (~6 h of embedding) |
