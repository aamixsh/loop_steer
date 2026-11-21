#!/usr/bin/env bash
# Move the Inkling 2 Mini work to another instance through GCS.
#   scripts/inkling_bundle.sh push   # on the source instance
#   scripts/inkling_bundle.sh pull   # on the new instance, from the repo root (main branch)
# Prefix: $LOOP_STEER_GCS/inkling_bundle (default $GCS_BUCKET/loop_steer; gcloud from envs/cloud.sh).
# Contents:
#   wheels/      the vLLM wheel built from vllm-project/vllm#69028 for sm_120 (RTX PRO 6000 Blackwell; other GPUs need a
#                rebuild, envs/inkling/README.md) -> /workspace/cache/wheels
#   runs.tar     data/runs/Inkling-2-Mini (generations, judge scores, directions, stream norms, activations, analysis,
#                capability runs) and the Inkling logs -> data/runs
# Model weights are never bundled: `pull` downloads the two pinned Inkling checkpoints from the Hugging Face Hub.
# `pull` then builds .venv-inkling (torch 2.14 cu130, the wheel, envs/inkling/requirements.lock.txt, this package) and
# runs an import check. The main .venv (judging, analysis) and the judge model come from scripts/setup_remote.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?push or pull}
source envs/cloud.sh  # gcloud on PATH (scripts/setup_remote.sh installs it), or the machine's cloud-sync env
BASE=${LOOP_STEER_GCS:-${GCS_BUCKET:?set GCS_BUCKET or LOOP_STEER_GCS}/loop_steer}
DEST=${BASE%/}/inkling_bundle
G() { gcloud ${GCLOUD_ACCOUNT:+--account=$GCLOUD_ACCOUNT} storage "$@"; }
export HF_HOME=${HF_HOME:-/workspace/cache/hf}
WHEELS=/workspace/cache/wheels

if [ "$MODE" = push ]; then
  G cp "$WHEELS"/vllm-*.whl "$DEST/wheels/"
  tar -cf - data/runs/Inkling-2-Mini $(ls data/runs/logs/inkling_* 2>/dev/null) | G cp - "$DEST/runs.tar"
  git rev-parse HEAD > /tmp/inkling_bundle_commit && G cp /tmp/inkling_bundle_commit "$DEST/COMMIT"
  G ls -l -r "$DEST" | tail -n 8
  echo "pushed -> $DEST"
  exit 0
fi

[ "$MODE" = pull ] || { echo "usage: $0 push|pull" >&2; exit 2; }
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/cache/uv} UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/workspace/cache/uv-python}
mkdir -p "$WHEELS" "$HF_HOME/hub" data/runs
G cp "$DEST/wheels/*.whl" "$WHEELS/"
G cat "$DEST/runs.tar" | tar -xf -
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from loop_steer.models import INKLING
for rid, rev in INKLING.items():
    print(snapshot_download(rid, revision=rev))
PY
[ -d .venv-inkling ] || uv venv .venv-inkling --python 3.12
uv pip install --python .venv-inkling/bin/python torch==2.14.0 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu130
grep -v "^vllm @" envs/inkling/requirements.lock.txt | grep -v "^torch==\|^torchvision==" > /tmp/inkling_reqs.txt
uv pip install --python .venv-inkling/bin/python "$WHEELS"/vllm-*.whl -r /tmp/inkling_reqs.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip install --python .venv-inkling/bin/python -e . --no-deps
source envs/inkling/env.sh
$PY_INKLING -c "import torch, vllm, transformers; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available(),
'capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None, '| vllm', vllm.__version__,
'| transformers', transformers.__version__)"
echo "bundle commit: $(G cat "$DEST/COMMIT")  checkout: $(git rev-parse HEAD)"
echo "Inkling ready: source envs/inkling/env.sh; see envs/inkling/README.md and docs/inkling.md"
