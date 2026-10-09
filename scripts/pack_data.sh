#!/usr/bin/env bash
# Bundle the run artifacts (and prompt CSVs) for moving to another machine.
#   scripts/pack_data.sh OUT.tar.zst|OUT.tar.gz [--with-activations] [--dry-run]   (.gz if the target has no zstd)
# Included: data/runs/<model>/{directions,generations,analysis,figures}, data/runs/logs, the prompt CSVs.
# Left out by default: activations (about 6.3 GB; only needed to recompute directions) and the stamped
# job/validation directories under data/runs. A .sha256 file is written next to the bundle.
# Never bundles credentials, caches or model weights (weights are re-downloaded at the pinned revisions).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=""; ACT=0; DRY=0
for a in "$@"; do
  case "$a" in
    --with-activations) ACT=1 ;;
    --dry-run) DRY=1 ;;
    *) OUT="$a" ;;
  esac
done
[ -n "$OUT" ] || { echo "usage: $0 OUT.tar.zst [--with-activations] [--dry-run]" >&2; exit 2; }

DATA=$(readlink -f data)
PROMPTS=$(.venv/bin/python -c "
from loop_steer import paths
paths._load_lab_profile()  # on shared servers the prompts may live under \$DATA_DIR
print(paths.dataset_dir())")
[ -f "$PROMPTS/train_harmful_prompts.csv" ] || { echo "prompt CSVs not found in $PROMPTS" >&2; exit 1; }

EXCL=(--exclude='runs/20[0-9][0-9]*--*' --exclude='*.lock')
[ "$ACT" = 1 ] || EXCL+=(--exclude='runs/*/activations')
ARGS=("${EXCL[@]}" -C "$DATA" runs -C "$(dirname "$PROMPTS")" "$(basename "$PROMPTS")")

echo "data:    $DATA/runs"
echo "prompts: $PROMPTS"
echo "activations: $([ "$ACT" = 1 ] && echo included || echo excluded)"
if [ "$DRY" = 1 ]; then
  tar -cf - "${ARGS[@]}" | wc -c | awk '{printf "uncompressed size: %.2f GB\n", $1/1e9}'
  exit 0
fi
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$OUT" in
  *.zst) tar -I 'zstd -T0 -3' -cf "$OUT" "${ARGS[@]}" ;;
  *.gz)  tar -czf "$OUT" "${ARGS[@]}" ;;
  *) echo "OUT must end in .tar.zst or .tar.gz" >&2; exit 2 ;;
esac
( cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )
ls -lh "$OUT" "$OUT.sha256"
echo "copy with:  rsync -avP $OUT $OUT.sha256 USER@HOST:DEST/"
