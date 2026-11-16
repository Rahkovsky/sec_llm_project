# Using the demo API

sec-intel can run as a small web API with a demo page, so the system can be shown live during a
presentation from any browser. It runs on the author's laptop with local models, is started
by hand for a demonstration, and is offline the rest of the time. It answers one question at a
time. Design and test results: [API_PLAN.md](API_PLAN.md).

## For the person trying it

You need two things from the presenter: the **address** (`https://<machine>.<tailnet>.ts.net`)
and the **API key** (a 32-character code; a new one is made each time the API starts).

### In a browser

1. Open the address. It works on a laptop or a phone, on any network.
2. Expand **API key** under the question box, paste the key, and click outside the box. The
   line above changes to "Expected time now: about … s". The browser remembers the key.
3. Type a question that names a company, for example:
   - *What does Waters say about its cyber insurance?*
   - *How much did Apple spend on share repurchases in fiscal 2024?*
   - *What does Costco say about its membership renewal rate?*

   If the company is not recognised, the page says so; type the ticker (e.g. `WAT`) in the
   **Ticker** field. The index covers S&P 500 companies' 10-K, 10-Q and 8-K filings from 2022
   on.
4. Choose a mode:
   - **Fast** (default), about 45–90 s: an answer with numbered sources. The independent judge
     is skipped, so the answer is marked unverified.
   - **Verified**, about 100–150 s: the judge also checks every claim against the filing and
     shows a verdict for each (SUPPORTED, PARTIAL, UNSUPPORTED). An answer that fails the check
     is refused.
5. Click **Ask**. The page shows a countdown and the step under way (In queue → Finding passages
   → Writing answer → Checking claims), then the answer and its sources: company, form, filing
   date, section, a link to the filing on SEC.gov, and the sentence of the filing that supports
   the answer.

If several people ask at once, questions wait in a queue; the countdown includes the wait.
The system answers only from the filings. When they do not contain the answer it says so
instead of guessing.

### From a terminal

```bash
K=<api key>; U=https://<machine>.<tailnet>.ts.net

# submit a question ("mode": "fast" or "verified"; "ticker" is optional)
curl -s -X POST $U/v1/questions -H "Authorization: Bearer $K" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does Costco say about its membership renewal rate?", "mode": "fast"}'
# -> {"id": "36a2d9ecd72f", "status": "queued", "estimated_seconds_at_submit": 54, ...}

# poll until "status" is "done" (or "failed"); the answer is in "result"
curl -s $U/v1/questions/36a2d9ecd72f -H "Authorization: Bearer $K"

# or follow it as server-sent events
curl -sN $U/v1/questions/36a2d9ecd72f/events -H "Authorization: Bearer $K"

# expected time for a new question right now
curl -s "$U/v1/estimate?mode=verified" -H "Authorization: Bearer $K"
```

| Endpoint | |
|---|---|
| `POST /v1/questions` | Submit `{"question", "ticker"?, "mode"?}`. `202` with `id`, queue position and estimated seconds; `422` if the question is longer than 500 characters or names no recognisable company; `429` if 20 questions are already waiting |
| `GET /v1/questions/{id}` | `status` (`queued`, `retrieving`, `answering`, `verifying`, `done`, `failed`, `cancelled`), `queue_position`, `estimated_seconds_remaining`, and `result` when done |
| `GET /v1/questions/{id}/events` | The same as server-sent events |
| `DELETE /v1/questions/{id}` | Cancel a question that is still queued |
| `GET /v1/estimate?mode=fast` | Expected time for a new question now |
| `GET /v1/health` | Server up, queue length (no key needed) |

The `result` holds `answer`, `abstained`, `verified`, `sources` (each with `n`, `company`,
`ticker`, `form`, `filing_date`, `section`, `url`, `excerpt`), `timings` per step,
`total_seconds` and `models`. A fast-mode result also has a `warning`; a verified one has
`claims` (each with `claim`, `verdict`, `source`) and `groundedness`.

### If something does not work

| Symptom | Cause |
|---|---|
| The page does not load | The API is not running, or it was started less than a minute ago. Reload after a minute |
| "missing or wrong API key" | The key was mistyped, or the API has been restarted since (new key) |
| "Could not tell which company the question is about" | No company recognised: add the ticker |
| The answer says the filings do not contain it | Retrieval did not find a passage with the answer; rephrase with the filing's wording or name the fiscal year |

## For the presenter

One-time setup: Ollama with the models, the index (README §5), the `api` extra
(`uv pip install -e ".[api]"`), and the Tailscale app with Funnel enabled for the tailnet (the
first `tailscale funnel 8000` prints the link that enables it). Give the machine a neutral
name in the Tailscale admin console: the address is written to public certificate logs.

```bash
scripts/serve_api.sh            # public address through Tailscale Funnel
scripts/serve_api.sh --local    # only http://127.0.0.1:8000
```

The script refuses to start while an evaluation is running (they share the models). It loads
the answer model, starts the API on `127.0.0.1:8000`, publishes it with Tailscale Funnel, and
prints the address and a new key. It keeps the Mac awake while running. **Ctrl-C** stops the
API and takes the address offline. Set `SEC_API_KEY` beforehand to reuse a key, and `CONFIG`
to use another config (default `config/local_16gb_v7_8b.yaml`).

Only the API is published. Ollama, the Phoenix tracing UI and everything else on the machine
stay unreachable from outside, and every request except `/v1/health` needs the key. Jobs and
their timings are kept in `data/api/jobs.db`; the time estimates start from the last 50 answered
questions there.
