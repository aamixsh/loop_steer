#!/usr/bin/env bash
# Set up a fresh GPU machine (vast.ai, RunPod, any Linux box) for loop-steer. Run from the repo root after cloning;
# docs/handoff.md is the step-by-step guide.
#   scripts/setup_remote.sh [--install-uv] [--cache-dir DIR] [--skip-models] [--no-sync] [--no-verify] [--no-gcloud]
# 1. checks the GPU driver, disk space and uv, 2. optionally points the big caches at DIR (see below),
# 3. installs the Google Cloud SDK (gcloud, for the GCS bundles and result pushes) into $LOOP_STEER_TOOLS (default
# /workspace/tools) if gcloud is missing, 4. `uv sync --locked`, 5. fetches the prompt CSVs at the pinned upstream
# commit if data/prompts is missing, 6. downloads the models at the pinned revisions, 7. runs
# scripts/verify_setup.sh --no-gpu. Logging in (gcloud, Hugging Face) stays manual: the script prints the commands.
# --cache-dir DIR  keep the Hugging Face cache (models, about 40 GB), the uv cache and the uv-managed Python under DIR
#                  (DIR/hf, DIR/uv, DIR/uv-python), e.g. a persistent volume (/workspace on vast.ai and RunPod) that
#                  several projects share. data/.cache/{huggingface,uv,uv-python} become symlinks to them, so jobs
#                  find the caches even in shells that did not export HF_HOME etc. (nohup, cron, agent tools).
#                  Without it everything stays inside the checkout (data/.cache, see scripts/env.sh).
# Do NOT run it while jobs use .venv: `uv sync` can change the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=1; SYNC=1; INSTALL_UV=0; VERIFY=1; CACHE_DIR=""; GCLOUD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-models) MODELS=0 ;; --no-sync) SYNC=0 ;; --install-uv) INSTALL_UV=1 ;; --no-verify) VERIFY=0 ;;
    --no-gcloud) GCLOUD=0 ;;
    --cache-dir) CACHE_DIR=${2:?--cache-dir needs a directory}; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -n "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR"; CACHE_DIR=$(cd "$CACHE_DIR" && pwd)
  export HF_HOME=$CACHE_DIR/hf UV_CACHE_DIR=$CACHE_DIR/uv UV_PYTHON_INSTALL_DIR=$CACHE_DIR/uv-python
  mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" data/.cache
  for pair in huggingface:hf uv:uv uv-python:uv-python; do
    link=data/.cache/${pair%%:*}; target=$CACHE_DIR/${pair##*:}
    if [ -L "$link" ]; then ln -sfn "$target" "$link"
    elif [ -e "$link" ]; then echo "note: $link already exists as a real directory; leaving it (move its contents to $target to share)"
    else ln -s "$target" "$link"; fi
  done
fi
source scripts/env.sh   # keeps preset HF_HOME / UV_CACHE_DIR / UV_PYTHON_INSTALL_DIR, otherwise data/.cache; temp in scratch/

echo "== GPU"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: need an NVIDIA driver (CUDA 12.8 or newer)" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "(the locked environment uses torch 2.10.0+cu128 and vLLM 0.18.0; needs a driver that supports CUDA 12.8)"

echo "== disk (need about 120 GB: models 40, uv cache + venv 30, run data 7-13, headroom)"
for d in . "${CACHE_DIR:-.}"; do
  avail=$(df -BG --output=avail "$d" | tail -1 | tr -dc 0-9)
  echo "free on $(df --output=target "$d" | tail -1): ${avail} GB"
  [ "$avail" -ge 120 ] || echo "WARNING: less than 120 GB free there"
done

echo "== gcloud"
source envs/cloud.sh
if command -v gcloud >/dev/null; then
  gcloud --version | head -1
elif [ "$GCLOUD" = 1 ]; then  # the SDK tarball into $LOOP_STEER_TOOLS; touches no shell profile
  tools=${LOOP_STEER_TOOLS:-/workspace/tools}; mkdir -p "$tools" "$CLOUDSDK_CONFIG"
  curl -fsSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
    | tar -xz -C "$tools"
  "$tools/google-cloud-sdk/install.sh" --quiet --usage-reporting=false --path-update=false --command-completion=false \
    >/dev/null
  gcloud --version | head -1
else
  echo "skipped (--no-gcloud)"
fi

echo "== uv"
if ! command -v uv >/dev/null; then
  if [ "$INSTALL_UV" = 1 ]; then  # installs uv into data/.cache/bin, touches no shell profile
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$LOOP_STEER_ROOT/data/.cache/bin" sh
  else
    echo "uv not found. Re-run with --install-uv (puts it in data/.cache/bin), or see https://docs.astral.sh/uv/" >&2
    exit 1
  fi
fi
uv --version
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

if [ "$VERIFY" = 1 ]; then echo "== verify"; scripts/verify_setup.sh --no-gpu || true; fi
cat <<EOF

Log in by hand (once per machine; credentials stay under /workspace):
  source envs/cloud.sh && gcloud auth login --no-launch-browser && gcloud storage ls \$GCS_BUCKET
  HF_HOME=\$HF_HOME .venv/bin/hf auth login && .venv/bin/hf auth whoami

Done. Next:
  source scripts/env.sh                       # in every new shell (or rely on the symlinks / exported variables)
  scripts/unpack_data.sh /path/to/bundle.tar.zst   # previous runs, if you have the bundle (see docs/handoff.md)
  scripts/verify_setup.sh --gpu 0             # GPU smoke test: HF and vLLM hooks on both looped models (~10 min)
EOF
if [ -n "$CACHE_DIR" ]; then
  cat <<EOF
To make the shared caches the default for every shell, add to ~/.bashrc:
  export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR
EOF
fi
