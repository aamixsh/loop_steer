#!/usr/bin/env bash
# Nemotron-Labs Dory 1.0 4B (looped nGPT) pipeline, one step per call (queue each step as its own gpuq job):
#   GPU=0 scripts/run_dory.sh validate     # our HF and vLLM implementations agree (greedy, clean / ablate / actadd)
#   GPU=0 scripts/run_dory.sh clean        # 500 train prompts x 3 CoTs x 3 answers, judged
#   GPU=0 scripts/run_dory.sh directions   # labels, activations at every site, directions, per-site analysis
# The layer screen comes after looking at the per-site analysis (scripts/run_select.sh dory TAG CANDIDATES...).
# Dory has no transformers or vLLM release support: loop_steer.dory (HF) and loop_steer.vllm_dory (NVIDIA's vLLM PR,
# ported to vLLM 0.18) are registered in-process. Same sampling as the other models (T=0.6, no top-p / top-k); CoT and
# answer caps 4096; 2 loops as trained. Steps skip what exists. Output goes to the step's log and to stdout.
set -u -o pipefail
cd "$(dirname "$0")/.."
STEP=${1:?step: validate|clean|directions}
PY=.venv/bin/python; G=${GPU:-0}; L=data/runs/logs; M=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
R=data/runs/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
exec 3>&1
run() { echo "+ $*" >&3; "$@"; }
tolog() { tee "$L/$1" >&3; }
mkdir -p $L

case $STEP in
  validate)  # layer 30: a recurrent attention sublayer (input 0-15, recurrent 16-55, output 56-71)
    run $PY scripts/validate_vllm_hooks.py --gpu $G --model $M --vllm-mem 0.5 --layer 30 2>&1 | tolog dory_validate_vllm_hooks.log ;;
  clean)
    [ -f $R/generations/train_clean.parquet ] || run $PY scripts/generate.py --gpu $G --model $M --split train \
        --n-prompts 500 --max-tokens 4096 --max-model-len 12288 --out train_clean.parquet 2>&1 | tolog dory_train_clean.log
    [ -f $R/generations/train_clean.scored.parquet ] || run $PY scripts/judge.py --gpu $G \
        $R/generations/train_clean.parquet 2>&1 | tolog judge_dory_train_clean.log ;;
  directions)
    run $PY scripts/build_sets.py $R/generations/train_clean.scored.parquet 2>&1 | tolog dory_build_sets.log
    [ -f $R/activations/train.pt ] || run $PY scripts/extract_activations.py --gpu $G --model $M \
        --labels $R/generations/labels.parquet --name train --labeled-only --max-batch-tokens 16384 2>&1 \
        | tolog dory_activations.log
    run $PY scripts/directions.py --model $M --activations train 2>&1 | tolog dory_directions.log
    for m in v4_cot v4_baseline; do
      run $PY scripts/analyze_directions.py --model $M --method $m 2>&1 | tolog dory_analyze_$m.log
    done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
