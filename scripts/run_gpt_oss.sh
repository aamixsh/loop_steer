#!/usr/bin/env bash
# gpt-oss-20b (MoE) pipeline, one step per call (queue each step as its own gpuq job):
#   GPU=0 scripts/run_gpt_oss.sh validate     # MoE weight edit = hook ablation? (HF + vLLM); HF vs vLLM hooks
#   GPU=0 scripts/run_gpt_oss.sh clean        # 500 train prompts x 3 CoTs x 3 answers, judged
#   GPU=0 scripts/run_gpt_oss.sh directions   # labels, activations at every layer, directions, per-site analysis
#   GPU=0 scripts/run_gpt_oss.sh screen       # 50 selection prompts: ablation per layer (as upstream: 7..21), random,
#                                             # actadd, weight edits with and without biases at upstream's layers
# Needs data/models/gpt-oss-20b-bf16 (scripts/prepare_gpt_oss_bf16.py). Steps skip what exists. Output goes to the
# step's log in data/runs/logs and to stdout. Same sampling as the other models (T=0.6, no top-p / top-k); CoT and
# answer caps 4096 (gpt-oss reasons at "Reasoning: medium", the template default).
set -u -o pipefail
cd "$(dirname "$0")/.."
STEP=${1:?step: validate|clean|directions|screen}
PY=.venv/bin/python; G=${GPU:-0}; L=data/runs/logs; M=openai/gpt-oss-20b; R=data/runs/gpt-oss-20b
exec 3>&1
run() { echo "+ $*" >&3; "$@"; }
tolog() { tee "$L/$1" >&3; }
mkdir -p $L

case $STEP in
  validate)
    DIR=${DIRECTION:+--direction $DIRECTION}  # e.g. DIRECTION=v4_baseline:l15 once directions exist; default random
    run $PY scripts/validate_moe_ortho.py --gpu $G $DIR --skip-vllm 2>&1 | tolog gptoss_validate_moe_ortho_hf.log
    run $PY scripts/validate_moe_ortho.py --gpu $G $DIR --skip-hf 2>&1 | tolog gptoss_validate_moe_ortho_vllm.log
    run $PY scripts/validate_vllm_hooks.py --gpu $G --model $M --vllm-mem 0.55 --layer 12 2>&1 \
        | tolog gptoss_validate_vllm_hooks.log ;;
  clean)
    [ -f $R/generations/train_clean.parquet ] || run $PY scripts/generate.py --gpu $G --model $M --split train \
        --n-prompts 500 --max-tokens 4096 --max-model-len 12288 --out train_clean.parquet 2>&1 | tolog gptoss_train_clean.log
    [ -f $R/generations/train_clean.scored.parquet ] || run $PY scripts/judge.py --gpu $G \
        $R/generations/train_clean.parquet 2>&1 | tolog judge_gptoss_train_clean.log ;;
  directions)
    run $PY scripts/build_sets.py $R/generations/train_clean.scored.parquet 2>&1 | tolog gptoss_build_sets.log
    [ -f $R/activations/train.pt ] || run $PY scripts/extract_activations.py --gpu $G --model $M \
        --labels $R/generations/labels.parquet --name train --labeled-only --max-batch-tokens 16384 2>&1 \
        | tolog gptoss_activations.log
    run $PY scripts/directions.py --model $M --activations train 2>&1 | tolog gptoss_directions.log
    for m in v4_cot v4_baseline; do
      run $PY scripts/analyze_directions.py --model $M --method $m 2>&1 | tolog gptoss_analyze_$m.log
    done ;;
  screen)
    C="none ablate:random:l15"
    for l in 7 9 11 13 15 17 19 21; do C="$C ablate:v4_cot:l$l ablate:v4_baseline:l$l"; done
    for l in 13 17; do for c in -1 -2; do C="$C actadd:v4_baseline:l$l:c=$c"; done; done
    # weight edits at upstream's layers (CoT 19, end-of-prompt 15), with and without the writers' biases
    C="$C ortho:v4_cot:l19 ortho:v4_cot:l19:bias=0 ortho:v4_baseline:l15 ortho:v4_baseline:l15:bias=0"
    run scripts/run_select.sh gptoss select_screen $C ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
