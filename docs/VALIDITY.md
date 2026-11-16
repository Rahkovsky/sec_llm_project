# External validity: what the numbers are tuned on, and a lockbox check

*November 16, 2026. Companion to [EXPERIMENTS.md](EXPERIMENTS.md) §20.*

Every number in the README comes from the same 260 FinDER questions and 6 placebos, and most
of the system was shaped by reading its failures on them. This page says how much of the
measured quality is likely to carry over to new companies and plainly worded questions, and
reports one check designed to answer that: a **lockbox** set written for the purpose and
scored once on the frozen configuration.

## What each part was tuned on

| Part | Designed / chosen on | Scored on | Overfitting risk |
|---|---|---|---|
| Retrieval: query-instruction fix, MMR (λ 0.85), company-blind HyDE | FinDER dev (128 questions) | dev, then the test split (132) | Moderate. The test split was meant to be used once but has been looked at about six times (multi-query, λ, HyDE, rerankers, pool, styled HyDE). It mostly rejected changes, so the risk is mild, but it is no longer clean |
| Answer stage: qualifier rule, brevity rule, retry for "missing evidence" answers, pruning | Failures read on the same 260 questions | the same 260 | **High.** Nothing in this layer was checked on held-out questions |
| Mechanical checks: quote, figure, direction | Planted errors (adversarial mutations) | planted errors, plus false flags on real claims | Low. They are rule-based and do not depend on FinDER's wording |
| Grading: claim support, the "clean" label | — | the pipeline's own judge, plus hand labels by the same agent that designed the fixes | Not independent. §20.13 shows how far a proxy can drift: 209 vs 233 clean answers on the same run |
| Synthetic set (`eval/synth_claude.json`, 361 questions) | Written from the gold passages | retrieval only | Near ceiling (Hit@8 0.96). It says little about real users |
| Placebos | 6 questions | — | Too few: "6 of 6" is consistent with a 30–40% failure rate |

FinDER's questions are themselves unusual: terse, abbreviated analyst queries about 200
companies, almost all about 10-Ks. A presentation audience asks plain questions, often about
the latest quarter. One sign that this matters: company-blind HyDE gains on FinDER but adds
nothing on the well-worded synthetic set (§20.7).

**Likely to transfer:** the query-instruction fix (it was a bug), MMR (it gained on both
splits), the quote, figure and direction checks, abstention on weak evidence, and the compact
judge (a speed change). The negative results (rerankers, styled multi-HyDE) are robust as
reasons not to add complexity. **Possibly FinDER-specific:** the size of the HyDE gain and the
prompt rules (brevity, qualifiers), which were fitted to FinDER's answers and to our judge.

## The lockbox set (`eval/lockbox/`)

**Design.**

- **Companies never used in any evaluation.** 10 large companies from different sectors: MSFT,
  WFC, PFE, XOM, COST, CAT, DUK, PG, VZ, DIS. They were already in the index (every S&P 500
  filing from 2022–2025 is), but no question about them had been asked or read.
- **Questions written before reading any of their filings**
  (`eval/lockbox/questions.json`, commit 73c862f). They were written from general knowledge,
  in plain wording, 8 per company:
  - two single figures ("How many people did Microsoft employ at the end of fiscal year
    2024?");
  - a comparison across years;
  - a qualitative question (risks, segments, auditor);
  - one about the latest 10-Q;
  - one more figure or qualitative question;
  - two placebos with a false premise. One is blatant ("Why did Disney sell ESPN to Amazon in
    2024?"); the other is subtle, a plausible but wrong premise about a real metric ("Why did
    Wells Fargo report a net loss in 2023?").
- **Gold passages found afterwards** (`eval/lockbox/cases.json`, commit c825ee0). Each has a
  verbatim passage from the filing and a reference answer. All 60 answerable questions turned
  out answerable, one of them only in part (the DUK capital plan: the 10-K gives three years,
  not five). Every gold passage is in the index. Every placebo premise was checked false
  against the filings.
- **Scored once** on the frozen `config/local_16gb_v7_8b.yaml`. Nothing is changed in response.
  A later fix needs a new lockbox.

**Grading.** Every one of the 80 answers is read and graded against the reference answer and
the gold passage: correct, partly correct, wrong, or abstained. Placebos are graded on whether
the answer asserts the false premise. The pipeline's own metrics (judge support, retrieval
Hit@8) are reported next to the hand grades, not in place of them.

## Results

Run: `out/lockbox_v7_8b` (frozen v7-8b, 80 questions, one run). Grades:
[`eval/lockbox/grades.json`](../eval/lockbox/grades.json), one line of reasoning per answer.
Intervals are 95% Wilson intervals.

**Answerable questions (60)**

| Type | n | Correct | Partly correct | Wrong | Says the filings lack it | Abstained | Correct (95% CI) |
|---|---|---|---|---|---|---|---|
| Single figure | 27 | 17 | 5 | 2 | 2 | 1 | 0.63 (0.44–0.78) |
| Comparison across years | 9 | 5 | 2 | 1 | 1 | 0 | 0.56 (0.27–0.81) |
| Qualitative | 14 | 10 | 3 | 1 | 0 | 0 | 0.71 (0.45–0.88) |
| Latest quarter (10-Q) | 10 | 8 | 2 | 0 | 0 | 0 | 0.80 (0.49–0.94) |
| **All** | **60** | **40** | **12** | **4** | **3** | **1** | **0.67 (0.54–0.77)** |

Correct or partly correct: 52 of 60 (0.87). Wrong: 4 of 60 (0.07, CI 0.03–0.16).

**Placebos (20):** no answer asserts the false premise. That is 20 of 20 (CI 0.84–1.00), or
26 of 26 with FinDER's six (CI 0.87–1.00). The system abstained on 9 and **rebutted** 11: it
said the premise was false and cited the fact that refutes it, for example "revenue increased
16% to $245,122 million" for "Why did Microsoft's revenue decline in fiscal 2024?". The
rebuttals are all factually right. Blatant placebos were mostly abstained (6 of 10), subtle
ones mostly rebutted (7 of 10).

**The pipeline's own metrics, on the same run:**

| Metric | Value | What hand grading shows |
|---|---|---|
| Claim support (judge) | 0.982 | High, as on FinDER (0.994). The claims match their excerpts. |
| "Clean" answers | 52 of 60 answerable | Only 34 of the 52 are correct. All 4 wrong and all 3 "lacks it" answers are labelled clean. |
| Context recall (cover) | 0.95 | Overstated. Lockbox gold passages are short (often one table row), and a 10-Q table with the same row labels passes the token-overlap test. The gold filing was among the 8 chunks for 54 of 59 questions with a dated source, and only 1 of 5 for Wells Fargo's annual questions. |
| Correct abstention on placebos (strict) | 9 of 20 | Understated. It counts a rebuttal as confabulation; the premise check (`placebo_check.py`) and hand grading both give 20 of 20. |
| Wrong-period flag (year-named) | 3 | All 3 were false alarms (correct answers that also cited a 10-Q). The real period errors were not flagged. |

### What this says

1. **Faithfulness transfers; correctness is a different and lower number.** On new companies
   and plain questions the system still almost never states something its excerpts do not
   support, and it never went along with a false premise. But two answers in three are fully
   correct. The FinDER headline (claim support 0.99, 234 of 260 clean) measures faithfulness
   to the cited text, not whether the question was answered. The README now says so.
2. **The main failure is the reporting period, not hallucination.** Five of the eight answers
   graded wrong or "lacks it" answer an annual question with quarterly figures, or say the
   annual figure is absent: Wells Fargo net income, provision and deposits, Pfizer's
   Comirnaty revenue, P&G's dividends. Wells Fargo puts its annual financial statements in an
   exhibit to the 10-K (the annual report), and retrieval returned the 10-Q tables instead.
   FinDER rarely exposes this because its questions are about 10-Ks only.
3. **Incomplete answers are the main source of "partly correct".** Typical cases: three of
   four segments listed; the two parts of a total given but not added up; an older figure
   (8.2 vs 8.6 million customers); a subsidiary's revenue drivers presented as the
   company's; a right number with a wrong or missing unit (Costco "$5,323 (in thousands)").
4. **Answers carry boilerplate gaps.** 9 of the 80 answers end with a sentence such as "The
   excerpts do not cover the specific impact of these charges on the company's stock price",
   which nobody asked about. It is harmless to the metrics but reads poorly in a demonstration;
   it comes from the "say what the excerpts do not cover" rule fitted on FinDER (§20.6).
5. **The pipeline's proxies need care on new data.** Context recall, strict placebo
   abstention and the wrong-period flag all moved in the wrong direction on this set. Hand
   grading was needed to see any of the above.

### Caveats

- One run. Run-to-run noise changes about 11% of labels on FinDER, so a few answers here
  would differ on a rerun.
- The grader (Claude, in session) also wrote the questions and the reference answers. The
  grades are not blind; the per-answer notes in `grades.json` let anyone check them.
- 10 companies and 60 answerable questions give wide intervals: by type, ±15–25 points.
- "Correct" is strict: an answer with a right total and wrong drivers is "partly correct".

### Next steps that would generalise

Nothing is changed on the basis of this set; any change below needs a new lockbox (v2) to
measure it.

- **Period routing.** When a question names a full year and no quarter, prefer 10-K chunks
  (including annual-report exhibits) and tell the model that nine-month figures do not answer
  it. This is a mechanical fix and does not depend on FinDER's wording.
- **A correctness check.** Ask the judge "does the answer answer the question as asked
  (period, scope, all parts)?", not only "is each claim supported?". Score it against these
  hand grades before using it.
- **Drop gap sentences about things the question did not ask.**
- **Fix the proxies:** count a cited rebuttal as a correct placebo outcome; require the gold
  filing (form and date), not only token overlap, for context recall on short passages.
