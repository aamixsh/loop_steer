#!/usr/bin/env bash
# Unpack a bundle made by scripts/pack_data.sh into the data directory (default: ./data, or $LOOP_STEER_DATA).
#   scripts/unpack_data.sh BUNDLE.tar.zst [DEST]
# Existing files are not overwritten (tar --keep-old-files), so it is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
BUNDLE=${1:?usage: $0 BUNDLE.tar.zst [DEST]}
DEST=${2:-${LOOP_STEER_DATA:-data}}
umask 077
if [ -f "$BUNDLE.sha256" ]; then
  ( cd "$(dirname "$BUNDLE")" && sha256sum -c "$(basename "$BUNDLE").sha256" )
else
  echo "warning: no $BUNDLE.sha256, skipping the checksum" >&2
fi
mkdir -p "$DEST"
tar -I zstd -xf "$BUNDLE" -C "$DEST" --keep-old-files
echo "unpacked into $DEST: $(ls "$DEST" | tr '\n' ' ')"
ls "$DEST/runs"
