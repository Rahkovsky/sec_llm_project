# Next steps: quality and speed ideas after v5

*November 14, 2026. Builds on [REPORT.md](REPORT.md) and [EXPERIMENTS.md](EXPERIMENTS.md) §16–§18.
Every idea below starts from a measured number, says what change it proposes, how it would be
measured, and roughly what it costs to try.*

## Where things stand

v5 (`config/local_16gb_v5.yaml`) is the recommended configuration. On the 260 FinDER questions,
each question's answer gets one label, either *clean* or its main failure:

| Outcome (v5) | Questions | Where the fault lies |
|---|---|---|
| Clean | 156 | – |
| Retrieval miss (abstained 25, answered from unrelated text 5) or wrong fiscal year (3) | 33 | Retrieval |
| Unsupported claims (31, plus 1 partial) | 32 | Answer step |
| Abstained although the evidence was retrieved | 26 | Answer step |
| Dropped qualifier (lexical flag) | 13 | Mostly the metric: only 2 of the 22 flagged claims are real |

A question takes about 104 s with `--workers 8` (122 s serially). Of the time spent in the models,
the judge takes 63%: it reads ~2,500 prompt tokens and writes ~520 tokens per question, at
~12 tokens/s. The answer step reads ~2,400 tokens at ~120 tokens/s. Model loads are only 7% of the
time, and Ollama already runs with flash attention and an 8-bit KV cache.

So the largest quality gains are in the answer step (58 questions) and retrieval (33). The largest
speed gain is in the judge.

---

## Quality

### Q1. Answer when the evidence is there (26 questions)

**What we see.** In 23 of the 26 "abstained with evidence" questions the answer model wrote only
sentences like "The excerpts do not describe …", although the retrieved chunks contain the gold
passage. FinDER questions are terse analyst phrases ("Impact on valuation & investor risk from
WAT's cyber ins."), and the model reads them too literally: no excerpt states an "impact on
valuation", so it reports nothing.

**Idea: quote first, then answer.** Ask the model to first copy the excerpt sentences that bear on
the question's topic, then write the answer from those quotes. Copying is easier than judging
relevance in one step, and an answer built from copied text also has fewer unsupported claims
(Q2). A cheaper alternative: when the answer contains only meta-claims, retry once with a prompt
that says "report what the excerpts do say about <topic>, even if they do not answer every part
of the question".

**Measure.** Rerun the 26 plus a random 40 control questions with v5 and the new prompt; count
abstentions with evidence, clean answers and claim support. Run-to-run noise changes ~11% of
labels, so compare against a v5 rerun of the same questions. **Cost:** a prompt change and two
~2-hour runs.

### Q2. Remove unsupported claims instead of shipping them (32 questions)

**What we see.** After the brevity rule, 44 claims in all 209 answers are unsupported (4.9%).
In 25 of the 31 answers labelled "unsupported claims", exactly one claim (of ~5.4) is
unsupported, so one sentence spoils the label.

**Idea A: prune after judging.** The judge already marks each claim. Drop the sentences whose
claims are unsupported, re-check that the remainder still answers the question, and show the user
the pruned answer. This is deterministic and adds no model call. The risk is a pruned answer that
reads oddly or loses context, so the pruned text needs a short readability check by hand on a
sample.

**Idea B: tell the model to drop what it cannot cite.** A prompt rule ("every sentence must end
with a citation; leave out anything you cannot cite"). This was the planned follow-up to the
brevity rule (issue #1).

**Measure.** Idea A can be scored on the existing v5 outputs with no new run: recompute labels
with unsupported sentences removed, then hand-check 20 pruned answers. **Cost:** a few hours of
code; no compute for the first estimate.

### Q3. Rank the right chunk into the top 8 (33 questions)

**What we see.** Hit@8 is 0.79 on the development questions. Adding every gold passage to the
index (v3 + gold) gained only 14 clean answers, so the gap is ranking, not missing documents
(issue #8). Cross-encoder rerankers all lowered Hit@8, probably because they scored the terse
question rather than the HyDE passage.

**Ideas, in order of cost:**

1. ~~**Rerank against the HyDE passage** (issue #5)~~ **Done, negative** (EXPERIMENTS §20.7):
   HyDE is a better reranker query than the question for all three models (+0.04–0.09 Hit@8),
   but no variant beats production. The best (RRF with Qwen@HyDE) is −0.057 on dev. On the
   well-worded synthetic set, Qwen on the question alone gains +0.025 near the ceiling. No
   reranker in the pipeline.
2. **Several HyDE passages:** generate 2–3 passages in different styles (risk factor, MD&A,
   notes to the financial statements) and average or fuse their rankings. FinDER answers sit in
   different sections, and one passage commits to one of them.
3. ~~**Larger candidate pool for MMR** (60 → 120) and a lab sweep of λ~~ **Done, no change**
   (EXPERIMENTS §20.11): at λ 0.85 a pool above 60 changes no list, and no λ beats 0.85 on
   dev. MMR is still worth +0.06 (dev) / +0.09 (test) over the plain top 8. The 14 dev gold
   chunks below dense rank 60 need a better query, not a different selection.

**Measure.** Paired retrieval labs with bootstrap CIs on dev, confirmed once on test, as for
every earlier retrieval change. **Cost:** an hour or two of compute per lab.

### Q4. Fix the measurements before trusting small gains

- **The dropped-qualifier label overstates.** The 13 questions labelled "dropped qualifier" come
  from a lexical flag; hand labels find only 2 real drops among 22 flags. A held-out label set
  for the judge check (issue #3) would let the report count real drops automatically.
- **FinDER ticker labels.** Three are wrong (§19). *Done (Nov 15):* `eval/finder_ticker_fixes.json`
  lists them; `eval/gold_plus_placebo_fixed.json` and `eval/finder_semantic_fixed.json` are the
  corrected copies; the originals are kept for comparability.
- **Judge agreement.** *Done in part (§20.9):* with planted errors, the judge catches 54 of 81
  (67% with the checks). It misses most direction flips, which the new direction check fixes for
  12 of 18 (v7), and a third of the added negations. Every support number comes from one judge (`qwen3:14b`). Re-judge a
  sample of 300 claims with a second model and with hand labels, to know the judge's error rate
  before the next prompt change is decided on a few points of support.
- **Noise.** With ~11% label churn between identical runs, decide answer-stage changes on
  paired reruns of the same questions, or average two runs (issue #4).

---

## Speed

Per question with 8 workers: ~60 s in the judge, ~27 s in the answer step, ~8 s in the HyDE
rewrite; retrieval runs while another question is in a model.

### S1. Make the judge write less (est. −20 to −30 s per question)

**What we see.** For every claim the judge writes back the full claim text, a verdict and the
supporting quote, about 100 tokens per claim at ~12 tokens/s.

**Idea.** Split the answer into numbered claims in code (one sentence each) and ask the judge to
return only `{"id": 3, "verdict": "SUPPORTED", "quote": "<at most 20 words>"}`. The deterministic
quote check still works on a short quote. That should cut the judge's output from ~520 to
~200 tokens.

**Measure.** Re-judge the existing v5 answers both ways and compare verdicts claim by claim
(agreement should be ≥ 95%) and time. **Cost:** a change in the verifier and a re-judging run of
about an hour.

### S2. Skip the judge for claims that are copied text (est. −5 to −15 s)

A claim whose words are nearly all in the cited chunk (high token overlap, no number or hedge
changed) can be marked supported without a model call. Only the rest go to the judge. The
dropped-qualifier check already looks for hedge words, so a copied claim that lost a "may" is
still caught. **Measure** with the same re-judging setup as S1: how many claims skip the judge,
and how many of those the judge would have marked unsupported (should be ~0).

### S3. Faster prompt processing (est. −5 to −10 s, uncertain)

**Done, negative** (EXPERIMENTS §20.8): the default batch of 512 is the fastest. With 1024 or
2048, qwen3:14b is 8–18% slower and gemma4:12b is no faster.

Ollama starts llama.cpp with a 512-token batch (`-b 512 -ub 512`). On Apple GPUs a larger batch
(`num_batch` 1024 or 2048) sometimes speeds up prompt processing. **Measure** with the 16-question
timing set (`out/timing/cases16.json`): ~30 minutes per setting. **Cost:** an option in the
provider.

### S4. Smaller judge (large saving, quality risk)

**Done** (EXPERIMENTS §20.9–20.10): with the figure and direction checks, `qwen3:8b` catches 68 of
81 planted errors against 69 for `qwen3:14b`. The full run gives 206 clean answers against v6's
209 (within noise), the same support and placebos, and 63 s of model time per question instead
of 81. `config/local_16gb_v7_8b.yaml` is now the recommended configuration. Its qualifier check
is not measurably weaker on the 14 hand-labelled drops (5 vs 6 found), but that set is small.

A judge of about 8B parameters would roughly double generation speed and fit next to the answer
model. It is only acceptable if it agrees with `qwen3:14b` and the hand labels on the claim sample
from Q4. That measurement comes first.

### S5. Interactive use

For `sec-intel ask`, the user waits for the HyDE rewrite, the answer and the judge in sequence
(~95 s). Show the answer as soon as it is written, marked "verifying", then update it with the
claim verdicts. On a 32 GB machine both models also stay loaded, which removes the swaps.

---

## Suggested order

| Step | Why first | Compute |
|---|---|---|
| Q2-A prune estimate on existing outputs | Free; tells whether pruning is worth building | none |
| S1 compact judge output | Largest speed gain; the re-judging set also serves S2 and S4 | ~1 h |
| Q4 judge-agreement sample | Needed before trusting the next small quality differences | ~1 h + hand labels |
| Q1 quote-first answering | Largest remaining quality gap (26 + part of 32 questions) | ~4 h |
| ~~Q3-1 rerank against HyDE~~ | Done: no gain (§20.7) | ~2 h |
| ~~Q3-3 MMR pool and λ~~ | Done: no change (§20.11) | minutes |
| ~~S4 smaller judge~~ | Done: `qwen3:8b` adopted, v7-8b (§20.10) | ~5 h |
| Full 266 run of the winners | Confirms on the whole set | ~6–7 h with S1 |
