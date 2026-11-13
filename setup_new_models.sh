#!/usr/bin/env bash
# Extra models for the experimental profiles config/local_qwen.yaml and config/local_gemma4.yaml.
# Run AFTER setup_local.sh. Needs Ollama >= 0.31 (brew upgrade ollama).
set -euo pipefail
ollama --version
ollama pull qwen3.6:35b-a3b   # ~22 GB, generator (qwen profile) / judge (gemma4 profile)
ollama pull gemma4:12b        # ~8 GB, judge (qwen profile)
ollama pull gemma4:26b        # ~16 GB, generator (gemma4 profile)
# The Qwen3-Embedding-4B weights (~8 GB) download from Hugging Face on first index/query.
echo "Tip: export OLLAMA_MAX_LOADED_MODELS=1 before starting Ollama on a 32 GB machine."
