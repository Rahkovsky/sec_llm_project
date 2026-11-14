# Plan: a question-answering API on this machine

*November 14, 2026. A plan, nothing built yet. For demonstrations during presentations,
started by hand; default mode fast. Timings are from the 16-question timing run
(EXPERIMENTS.md §16) on the M1 Pro, 16 GB.*

## What the API has to cope with

- **One question at a time.** Only one of the two models (generator `gemma4:12b`, judge
  `qwen3:14b`) fits in memory, and Ollama serves one request at a time. A second caller
  waits; the API must queue questions and say how long the wait is.
- **Slow answers.** A verified answer takes ~120 s (HyDE 11 s, retrieval 13 s, answer 30 s,
  judge 68 s). That is too long for one blocking HTTP request through a tunnel or proxy,
  so the API is asynchronous: submit, then poll or stream.
- **Two modes.** *Verified*: the judge checks every claim, and an answer that fails the
  check is refused (~120 s; ~95 s once the compact judge of §20.2 is adopted). *Fast*: no
  judge, so no claim verdicts and no refusal on weak support (~55 s). Fast answers must be
  marked unverified.
- **Shared machine.** Batch evaluations use the same models; they are paused by hand while
  the API is up.

## Design

### Endpoints (FastAPI, versioned under `/v1`)

| Method and path | Purpose |
|---|---|
| `POST /v1/questions` | Submit `{question, ticker?, mode: "verified" \| "fast"}`. Returns `202` with `id`, `queue_position`, `estimated_seconds`, and the poll and stream URLs. `429` when the queue is full |
| `GET /v1/questions/{id}` | Status (`queued`, `retrieving`, `answering`, `verifying`, `done`, `failed`, `cancelled`), current stage, updated `estimated_seconds_remaining`, and the result when done |
| `GET /v1/questions/{id}/events` | Server-sent events: stage changes, then the answer, then (verified mode) the claim verdicts |
| `DELETE /v1/questions/{id}` | Cancel a queued question |
| `GET /v1/estimate?mode=verified` | Expected time for a new question now, without submitting one |
| `GET /v1/health` | Index loaded, Ollama reachable, queue length, paused or not |

**Result body:** answer text; citations (company, form, filing date, section, SEC URL,
quote); `abstained` and its reason; `mode`; in verified mode the per-claim verdicts and
groundedness; the resolved ticker and how it was resolved; per-stage timings; model names.
A fast-mode result carries `"verified": false` and a one-line warning.

### Expected time

The pipeline already logs every model call with its load, prompt and generation time
(`providers.call_log`). The API keeps a rolling median per stage and mode over the last
~50 questions (seeded with the timing-run numbers), and estimates:

```
wait  = sum over the questions ahead of (their mode's median total)
own   = median per stage for this mode, minus the model load if that model is already loaded
total = wait + own
```

While a question runs, the remaining time is recomputed from the stages left. The estimate
is reported with its basis (`"based_on": 50`). After launch, the logged error of the
estimates shows whether medians are good enough or need the answer length as a factor.

### Inside the server

- **One worker thread** owns the `SECIntelPipeline`; the index, embedder and entity
  resolver load once at startup (FastAPI lifespan). Requests only enqueue. This matches the
  hardware, and the pipeline does not need to be thread-safe.
- **Mode switch per question:** fast mode runs the same pipeline with the verifier turned
  off for that call (`ask(..., verify=False)`, a small change in `GroundedAnswerer`), so
  the index is not loaded twice.
- **Ordering:** first in, first out. A later option: serve waiting fast-mode questions
  together while the generator is loaded (the `ModelScheduler` idea), which avoids judge
  swaps.
- **Persistence:** jobs and results in SQLite (`data/api/jobs.db`), so a restart keeps
  results; results kept 7 days.
- **Limits:** queue at most 20 questions; question at most 500 characters; a timeout of
  10 minutes per question.

### Reaching it from the internet

Use: demonstrations during a presentation, started by hand when needed, not a service for
clients. Do not open a port on the router: a tunnel makes an outbound connection from this
machine, so nothing listens on the internet directly. Prices as of November 2026:

| Option | Cost | URL | Fit for a demo |
|---|---|---|---|
| **Tailscale Funnel** (recommended) | Free (Personal plan, non-commercial use) | Stable: `https://<machine>.<tailnet>.ts.net` | Same URL every time, so it can go on a slide. Needs a Tailscale account and the Tailscale app (App Store or standalone build on macOS); bandwidth is capped, which is irrelevant for text answers. Funnel is still labelled beta |
| Cloudflare quick tunnel | Free, no account | Random `https://<words>.trycloudflare.com`, new on every start | Zero setup (`brew install cloudflared`; one command). No uptime guarantee; the URL must be read off the terminal each time, so not printable on a slide |
| Cloudflare named tunnel | Free tunnel; needs a domain on Cloudflare, about $11/year for a `.com` at Cloudflare's at-cost price | Your own, e.g. `https://sec.example.com` | Most polished, and Cloudflare Access can add an email login. Only worth it if a custom domain matters |
| ngrok | Free plan: one static `ngrok-free` domain; paid plans from about $8–10/month | Stable on the free plan | Browser visitors on the free plan first see an ngrok warning page (API calls with a header skip it); limits on the free plan are tight |

Security, whatever the tunnel:

- **One API key** for the demo (`Authorization: Bearer ...`), set in an environment
  variable when the server starts; a request without it is refused. Rotate it after the
  presentation.
- Uvicorn listens on `127.0.0.1` only; Ollama stays on `127.0.0.1:11434`; Phoenix tracing
  (port 6006) is not tunnelled.
- Question length limit (500 characters) and queue limit (20), so a leaked URL cannot tie
  the machine up for hours.
- Logs keep the question, timings and outcome only.

### Starting and stopping

No always-on service: one script (`scripts/serve_api.sh`) checks that no evaluation is
running, starts Ollama's models warm (one short request each, so the first demo question
does not pay the load time), the API and the tunnel, prints the public URL and the key, and
stops all of it on Ctrl-C. `caffeinate` keeps the Mac awake while it runs. Evaluations are
paused by hand while the API is up, so the pause-lock is dropped from the plan.

### Demo page

For a presentation a page is easier to show than `curl`: the API also serves one static
page at `/` with a question box, a mode switch (fast by default), a countdown from the
estimate, the stages as they happen, and the answer with its citations and, in verified
mode, the claim verdicts. It calls the same `/v1` endpoints with the key.

## Steps

| Step | Work | Done when |
|---|---|---|
| 1. Pipeline hooks | `ask(..., verify=False)`; a stage callback so the worker can report `answering` / `verifying`; the call log returned with the answer | Unit tests pass with the offline config |
| 2. API core | FastAPI app (`src/sec_llm_project/sec_intel/app/api.py`), job queue, worker thread, SQLite store, `POST`/`GET`/`DELETE`, `/health`; `api` extra in `pyproject.toml` (`fastapi`, `uvicorn`); default mode `fast` | Tests with FastAPI's `TestClient` and the mock config; a real question answered locally in both modes |
| 3. Estimates | Rolling stage medians, `/v1/estimate`, remaining-time updates; log estimate vs actual | 20 real questions: median absolute error of the estimate under 20% |
| 4. Streaming and demo page | SSE events; the page at `/` | The page shows the countdown, the stages and the answer |
| 5. Key and limits | Bearer key, size and queue limits | Requests without the key are refused |
| 6. Internet | Tunnel of your choice, `scripts/serve_api.sh` | The page works from a phone off the home network |
| 7. Docs | README section with the start command, example `curl` calls and the response schema | – |

## Decisions

Made (Nov 14): demonstrations only, started by hand; evaluations paused by hand while it
runs; default mode **fast**.

Open: the tunnel. Recommended: Tailscale Funnel (free, stable URL). Alternative with no
account at all: a Cloudflare quick tunnel (random URL each time).
