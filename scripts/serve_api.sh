#!/bin/bash
# Start the question-answering API for a demonstration (docs/API_PLAN.md).
#
#   scripts/serve_api.sh            # API on 127.0.0.1:8000 + public URL via Tailscale Funnel
#   scripts/serve_api.sh --local    # API on 127.0.0.1:8000 only
#
# Ctrl-C stops the API and takes the public URL offline. The demo key is printed at
# start; set SEC_API_KEY beforehand to reuse a key, otherwise a new one is generated.
# Environment: CONFIG (default config/local_16gb_v6.yaml), PORT (default 8000).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-config/local_16gb_v6.yaml}"
PORT="${PORT:-8000}"
PUBLIC=1
[ "${1:-}" = "--local" ] && PUBLIC=0

if pgrep -f "tools/provenance_dag.py|tools/judge_lab.py|tools/update_index.py" >/dev/null; then
  echo "An evaluation or index job is running; stop it first (it shares the models)." >&2
  exit 1
fi
if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Ollama is not running (brew services start ollama)." >&2
  exit 1
fi
if [ "$PUBLIC" = 1 ] && ! command -v tailscale >/dev/null; then
  echo "tailscale CLI not found (Tailscale app: Settings > Install CLI), or use --local." >&2
  exit 1
fi

export SEC_API_KEY="${SEC_API_KEY:-$(openssl rand -hex 16)}"
# the embedding models are cached locally; do not call the Hugging Face Hub at start
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Warm the generator (fast mode uses only it), so the first question pays no load time.
GEN_MODEL=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'src')
from sec_llm_project.sec_intel.core.config import AppConfig
print(AppConfig.load('$CONFIG').llm.model)")
echo "Loading $GEN_MODEL ..."
curl -sf http://127.0.0.1:11434/api/generate \
  -d "{\"model\": \"$GEN_MODEL\", \"prompt\": \"ok\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
  >/dev/null || echo "warning: could not warm $GEN_MODEL" >&2

cleanup() {
  [ "$PUBLIC" = 1 ] && tailscale funnel reset >/dev/null 2>&1 || true
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
  echo "Stopped."
}
trap cleanup EXIT INT TERM

.venv/bin/sec-intel --config "$CONFIG" serve --host 127.0.0.1 --port "$PORT" &
API_PID=$!
echo "Starting the API (loading the index takes ~30 s) ..."
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/health" >/dev/null && break
  kill -0 "$API_PID" 2>/dev/null || { echo "The API failed to start." >&2; exit 1; }
  sleep 1
done

URL="http://127.0.0.1:$PORT"
if [ "$PUBLIC" = 1 ]; then
  tailscale funnel --bg "$PORT" >/dev/null
  URL="https://$(tailscale status --json | .venv/bin/python -c \
    'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
fi

cat <<EOF

  Demo page:  $URL/
  API key:    $SEC_API_KEY
  Example:    curl -s -X POST $URL/v1/questions -H "Authorization: Bearer $SEC_API_KEY" \\
                -H 'Content-Type: application/json' \\
                -d '{"question": "What does Waters say about its cyber insurance?"}'

  Ctrl-C to stop$( [ "$PUBLIC" = 1 ] && echo " (the public URL goes offline too)").
EOF

# keep the Mac awake while the API runs
caffeinate -dims -w "$API_PID"
