#!/usr/bin/env bash
# Copy the small summary tables of the runs into results/ (tracked in git) so results can be read and compared
# without the data bundle:  data/runs/<model>/analysis/*.csv -> results/<model>/, plus the capability summary.
# Only metrics are exported: no generations, logs, prompts, activations or weights (data/ stays out of git).
#   scripts/export_results.sh          then:  git add results && git commit
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=$(readlink -f data)/runs
for dir in "$RUNS"/*/analysis; do
  [ -d "$dir" ] || continue
  model=$(basename "$(dirname "$dir")")
  mkdir -p "results/$model"
  cp -f "$dir"/*.csv "results/$model/" 2>/dev/null || true
done
[ -f "$RUNS/capability_summary.csv" ] && cp -f "$RUNS/capability_summary.csv" results/
find results -type f | sort | sed 's/^/exported /'
du -sh results | awk '{print "total", $1}'
