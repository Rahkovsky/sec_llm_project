#!/bin/bash
# Replace amended (10-K/A) and Exhibit-13-wrapper 10-Ks with the real annual
# report, and add filings from 2022 (fiscal 2021), for every ticker in data/input/10-K.
# Needs SEC_USER_NAME / SEC_USER_EMAIL (EDGAR identity). Old files are moved, not
# deleted, to data/input_replaced/10-K/. Resumable: re-run on interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SEC_USER_EMAIL:?export SEC_USER_NAME and SEC_USER_EMAIL first (see docs/RUNBOOK.md)}"
mkdir -p data/input_replaced/10-K
while read -r rel; do
  [ -f "data/input/10-K/$rel" ] || continue
  t=${rel%%/*}; mkdir -p "data/input_replaced/10-K/$t"
  mv "data/input/10-K/$rel" "data/input/10-K/${rel%.txt}.json" "data/input_replaced/10-K/$t/" 2>/dev/null || true
  yr=20$(echo "$rel" | sed -E 's/.*-([0-9]{2})-[0-9]+\.txt/\1/')
  .venv/bin/python - "$t|10-K|$yr" <<'PY'
import json, sys
p = "data/input/.fetch_checkpoint.json"
d = json.load(open(p)); d.pop(sys.argv[1], None); json.dump(d, open(p, "w"))
PY
done < tools/refetch_10k_list.txt
tickers=$(ls data/input/10-K | grep -v '^\.' | tr '\n' ' ')
.venv/bin/sec-intel fetch $tickers --forms 10-K --years 2022-2025
echo "replaced: $(find data/input_replaced/10-K -name '*.txt' | wc -l) | 10-K files now: $(ls data/input/10-K/*/*.txt | wc -l)"
touch data/input/.refetch_10k_done
