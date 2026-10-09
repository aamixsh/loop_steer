#!/usr/bin/env bash
# Set up a fresh machine for loop-steer. Run from the repo root after cloning.
#   scripts/setup_remote.sh [--skip-models] [--no-sync]
# 1. checks the GPU driver and uv, 2. `uv sync --locked`, 3. fetches the prompt CSVs at the pinned upstream
# commit if data/prompts is missing, 4. downloads the models at the pinned revisions, 5. checks that vLLM sees the GPUs.
# Do NOT run this on a machine where jobs already use .venv: it can change the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=1; SYNC=1
for a in "$@"; do
  case "$a" in --skip-models) MODELS=0 ;; --no-sync) SYNC=0 ;; *) echo "unknown option $a" >&2; exit 2 ;; esac
done
umask 077

echo "== GPU"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: need an NVIDIA driver (CUDA 12.8 or newer)" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "(the locked environment uses torch 2.10.0+cu128 and vLLM 0.18.0; needs a driver that supports CUDA 12.8)"

echo "== uv"
command -v uv >/dev/null || { echo "uv not found: install it first, see https://docs.astral.sh/uv/" >&2; exit 1; }
if [ "$SYNC" = 1 ]; then uv sync --locked; fi

echo "== prompts"
if [ -f data/prompts/train_harmful_prompts.csv ] || [ -n "${LOOP_STEER_PROMPTS:-}" ]; then
  echo "already present"
else
  tmp=$(mktemp -d)
  git clone -q https://github.com/kureha-yamaguchi/reasoning-manipulation "$tmp/rm"
  git -C "$tmp/rm" checkout -q 56a763d837cc63f30a13f830ad4ebd05890c7455
  mkdir -p data/prompts
  cp "$tmp"/rm/dataset/{train,test}_{harmful,harmless}_prompts.csv data/prompts/
  echo 56a763d837cc63f30a13f830ad4ebd05890c7455 > data/prompts/SOURCE_COMMIT
  rm -rf "$tmp"
  echo "fetched into data/prompts"
fi

echo "== models"
if [ "$MODELS" = 1 ]; then .venv/bin/python scripts/prefetch_models.py; else echo "skipped"; fi

echo "== check"
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "gpus", torch.cuda.device_count())
import vllm, transformers
print("vllm", vllm.__version__, "transformers", transformers.__version__)
from loop_steer.cot import load_prompts
print("test prompts:", len(load_prompts("test")), "train prompts:", len(load_prompts("train")))
PY
echo "done. Unpack the data bundle with scripts/unpack_data.sh if you have one."
