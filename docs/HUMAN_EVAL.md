# Human evaluation & judge validation

Automated faithfulness scores are **themselves unvalidated**. A lexical judge
measures keyword overlap; an LLM judge measures entailment — but neither has been
checked against human judgement on *this* corpus. This protocol produces a small
human-graded sample that serves two purposes:

1. **Validate the automated judge** — measure how well its verdicts agree with
   humans, so we know how much to trust `mean_groundedness` / `lexical_groundedness`.
2. **Capture what the automated metrics structurally cannot** — *correctness*
   (is the answer true?) and *richness* (is it complete and specific, or a thin
   answer that games the metric?). See [FUTURE_WORK.md](FUTURE_WORK.md) §4–5.

---

## Why we need humans here

- **Groundedness ≠ correctness.** The judge checks the answer against the
  *retrieved* chunks, not against the truth. A confidently wrong answer grounded
  in a wrongly-retrieved chunk scores high. Only a human (or a gold answer) closes
  this gap.
- **Cheap answers game overlap metrics.** A short, vague answer that echoes
  evidence keywords can outscore a precise, complete one. We explicitly reward
  *richness* so the system isn't optimised toward thin, evasive answers.
- **The judge itself can be wrong.** Without a human anchor, an LLM judge's
  groundedness number is an opinion, not a measurement.

---

## The rubric (each axis 0–2)

| Axis | 0 | 1 | 2 |
| --- | --- | --- | --- |
| **Faithfulness** | Claims contradict or are absent from the cited evidence | Partially grounded; some claims unsupported | Every claim is supported by the cited evidence |
| **Correctness** | Wrong vs. the gold answer / filing | Partially correct or incomplete | Correct |
| **Citation quality** | Citations missing, wrong, or irrelevant | Some relevant citations, imprecise | Precise and sufficient to verify the answer |
| **Richness** | Trivial / evasive / one-liner that dodges the question | Adequate but thin | Complete, specific, and responsive |

Plus one flag:

- **abstention_appropriate** — `true` if the question was genuinely
  unanswerable from the index (or evidence was too weak) and the system correctly
  abstained; `false` if it abstained when it should have answered, or answered
  when it should have abstained; `n/a` otherwise.

**Faithfulness vs. correctness is the key distinction.** An answer can be faithful
(grounded in its citations) yet incorrect (the citations were the wrong evidence).
Score them independently — their *divergence* is the signal we most want.

---

## Procedure

1. **Generate the sheet** (real answers + citations + gold evidence, blank scores):
   ```
   python tools/dump_eval_outputs.py --eval eval/financebench_2023.json --load-index --sample 13 --out out/human_eval_fb
   ```
   Writes `out/human_eval_fb.jsonl` (re-importable) and `out/human_eval_fb.md`
   (human-readable). FinanceBench is the best starting set because it ships gold
   answers, which anchor the *correctness* axis.

2. **Score** each item in the `.md` sheet using the rubric. Record the four 0–2
   scores and the abstention flag back into the `scores` object of each record in
   the `.jsonl` (mirror the blanks in `SCORE_TEMPLATE`).

3. **Two raters minimum** for any axis you plan to report — compute inter-rater
   agreement (Cohen's κ) so the human anchor is itself trustworthy. Adjudicate
   disagreements.

4. **Validate the automated judge.** For each answered item, compare the human
   *faithfulness* (0–2) against the judge's per-item groundedness:
   - Lexical judge today → expect weak agreement, especially on numeric questions.
   - When an LLM judge is enabled (see below) → re-measure; agreement should rise.
   Report the agreement (κ or Spearman) alongside any faithfulness number so it is
   never quoted as if self-evidently trustworthy.

---

## Enabling a real (independent) LLM judge

The automated judge must never be the generator grading itself — this is enforced
at startup (`build_verifier` raises on a self-grading judge). For the local stack:

```
ollama pull mistral:7b
```
then set `verification.enabled: true` in `config/local.yaml` (judge is already
wired to `mistral:7b`, a different family from the `gemma3` generator). For cloud,
configure a cross-vendor judge (e.g. generator = OpenAI, judge = Anthropic).

Re-run step 1 with verification on; the per-answer `verification` report is then
produced by the LLM judge, and step 4 measures its agreement with the humans.

---

## What to report

- Human means per axis (faithfulness / correctness / citation / richness).
- **Faithfulness − correctness gap** (how often grounded-but-wrong occurs).
- Judge agreement (human faithfulness vs. automated groundedness), with κ/ρ.
- Abstention precision/recall from the abstention flag.

Keep the graded `.jsonl` under `eval/human/` as a growing, versioned gold set;
re-run the agreement check whenever the judge model or prompts change.
