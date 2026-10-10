#!/usr/bin/env bash
# Check that this machine is ready to run or resume loop-steer work. Prints PASS / WARN / FAIL per check and exits 1
# if anything FAILed.
#   scripts/verify_setup.sh [--no-gpu] [--gpu N]
# Without a flag the CPU checks run, then the GPU smoke test on GPU 0 (needs the GPU free: vLLM takes 30% of it).
#   --no-gpu   CPU checks only (environment, pins, prompts, models in the cache, saved results re-summarize)
#   --gpu N    GPU smoke test on GPU N: HF hooks on Ouro and Nanbeige (scripts/loop_smoke.py) and vLLM hooks against
#              Hugging Face on Ouro (scripts/validate_vllm_hooks.py), about 10 minutes in total
cd "$(dirname "$0")/.."
source scripts/env.sh
PY=.venv/bin/python
GPU=0; DO_GPU=1
while [ $# -gt 0 ]; do
  case "$1" in --no-gpu) DO_GPU=0 ;; --gpu) GPU=${2:?}; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac
  shift
done
FAILS=0
ok()   { echo "PASS  $*"; }
warn() { echo "WARN  $*"; }
bad()  { echo "FAIL  $*"; FAILS=$((FAILS + 1)); }

# 1. environment and pins
if [ ! -x $PY ]; then bad ".venv missing: run uv sync --locked"; exit 1; fi
VERS=$($PY -c "import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)" 2>&1 | tail -1)
case "$VERS" in
  "2.10.0+cu128 0.18.0 4.57.6") ok "environment: torch/vllm/transformers = $VERS" ;;
  *) bad "environment: got '$VERS', expected '2.10.0+cu128 0.18.0 4.57.6' (uv sync --locked)" ;;
esac
$PY -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>/dev/null && ok "python 3.12" || bad "python is not 3.12"

# 2. prompts
CNT=$($PY -c "
from loop_steer.cot import load_prompts
print([len(load_prompts(s, k)) for s in ('train', 'test') for k in ('harmful', 'harmless')])" 2>&1 | tail -1)
[ "$CNT" = "[1459, 1174, 487, 250]" ] && ok "prompts: $CNT (train/test x harmful/harmless)" || bad "prompts: got $CNT, expected [1459, 1174, 487, 250]"

# 3. models resolve from the cache at the pinned revisions (offline: nothing is downloaded)
if HF_HUB_OFFLINE=1 $PY scripts/prefetch_models.py > $TMPDIR/loop_steer_prefetch.$$ 2>&1; then
  ok "models in the Hugging Face cache at the pinned revisions ($(HF_HOME=${HF_HOME:-} $PY -c 'import os;print(os.environ.get("HF_HOME") or "data/.cache/huggingface")'))"
else
  bad "models missing from the cache (run scripts/prefetch_models.py); $(tail -1 $TMPDIR/loop_steer_prefetch.$$)"
fi
rm -f $TMPDIR/loop_steer_prefetch.$$

# 4. vLLM IPC socket path (a Unix socket path is limited to 107 characters)
LEN=$($PY -c "
from loop_steer.paths import SCRATCH_ROOT
print(len(str(SCRATCH_ROOT.resolve() / 'ipc-00000000')) + 1 + 36)")
[ "$LEN" -le 107 ] && ok "vLLM socket path length $LEN <= 107" || warn "vLLM socket path would be $LEN > 107; setup_job_env falls back to the system temp dir"

# 5. saved results re-summarize to the published numbers (needs the data bundle)
F=data/runs/Ouro-1.4B-Thinking/generations/test_hooks
if [ -f $F/ablate_v4_baseline_t3.l16.scored.parquet ]; then
  R=$($PY scripts/summarize.py $F --cont ${F}_ext 2>/dev/null | awk '$1=="ablate_v4_baseline_t3.l16"{print $(NF-2)}')
  case "$R" in 0.10[0-9]) ok "Ouro hook ablation (test set) re-summarizes to all-refusal $R (README: 10%)" ;;
    *) bad "Ouro test-set summary gave '$R', expected about 0.104" ;; esac
else
  warn "no run data in data/runs (unpack the bundle: scripts/unpack_data.sh); skipping the re-summary check"
fi

# 6. GPU
if [ "$DO_GPU" = 1 ]; then
  command -v nvidia-smi >/dev/null && ok "GPUs: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tr '\n' ';')" \
    || bad "nvidia-smi not found"
  USED=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
  [ "${USED:-0}" -lt 2000 ] && ok "GPU $GPU is free" || { bad "GPU $GPU already holds ${USED} MiB; free it or pass --gpu"; exit 1; }
  export HF_HUB_OFFLINE=1
  for M in ByteDance/Ouro-1.4B-Thinking Nanbeige/Nanbeige4.2-3B; do
    if $PY scripts/loop_smoke.py --gpu $GPU --model $M > $TMPDIR/loop_steer_smoke.$$ 2>&1; then
      ok "HF hooks on $M: $(grep -c '^\[' $TMPDIR/loop_steer_smoke.$$) smoke sections ran"
    else bad "loop_smoke.py failed for $M: $(tail -2 $TMPDIR/loop_steer_smoke.$$ | tr '\n' ' ' | cut -c1-200)"; fi
  done
  if $PY scripts/validate_vllm_hooks.py --gpu $GPU --model ByteDance/Ouro-1.4B-Thinking --layer 12 --vllm-mem 0.3 \
      > $TMPDIR/loop_steer_vh.$$ 2>&1; then
    CLEAN=$(awk '$1=="clean"{print $2}' $TMPDIR/loop_steer_vh.$$); ABL=$(awk '$1=="ablate"{print $2}' $TMPDIR/loop_steer_vh.$$)
    awk -v c="$CLEAN" -v a="$ABL" 'BEGIN{exit !(c >= 0.85 && a >= 0.85)}' \
      && ok "vLLM hooks match Hugging Face on Ouro (clean $CLEAN, ablate $ABL; reference 0.9-1.0)" \
      || bad "vLLM vs HF agreement low on Ouro (clean $CLEAN, ablate $ABL; expected >= 0.85)"
  else bad "validate_vllm_hooks.py failed: $(tail -2 $TMPDIR/loop_steer_vh.$$ | tr '\n' ' ' | cut -c1-200)"; fi
  rm -f $TMPDIR/loop_steer_smoke.$$ $TMPDIR/loop_steer_vh.$$
fi

echo; [ $FAILS -eq 0 ] && echo "all checks passed" || { echo "$FAILS check(s) FAILED"; exit 1; }
