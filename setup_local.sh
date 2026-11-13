#!/usr/bin/env bash
# Recreate the sec_llm_project environment (follows docs/INSTALLATION.md). Run on your Mac from the repo root.
# Usage: SEC_USER_NAME="Your Name" SEC_USER_EMAIL="you@example.com" ./setup_local.sh [--full]
set -euo pipefail
: "${SEC_USER_NAME:?set SEC_USER_NAME}"; : "${SEC_USER_EMAIL:?set SEC_USER_EMAIL}"
export SEC_USER_NAME SEC_USER_EMAIL

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev,yaml,embeddings,vectorstore,sec]"

command -v ollama >/dev/null || brew install ollama
brew services start ollama || true
sleep 3
ollama pull gemma3:12b
ollama pull mistral:7b   # optional independent judge

sec-intel fetch MSFT TSLA --forms 10-K 10-Q 8-K --years 2023-2025
sec-intel --config config/local.yaml index data/input/10-K/MSFT data/input/10-K/TSLA
# Full corpus (~10h): python tools/fast_index.py data/input/10-K data/input/10-Q data/input/8-K --max-chars 3600 --batch-size 16
echo "Done. Try: sec-intel --config config/local.yaml monitor MSFT 2024 2025"
