#!/usr/bin/env bash
# Nemotron-Labs Dory 1.0 4B (looped nGPT) pipeline, one step per call (queue each step as its own gpuq job):
#   GPU=0 scripts/run_dory.sh validate     # our HF and vLLM implementations agree (greedy, clean / ablate / actadd)
#   GPU=0 scripts/run_dory.sh clean        # 500 train prompts x 3 CoTs x 3 answers, judged
#   GPU=0 scripts/run_dory.sh directions   # labels, activations at every site, directions, per-site analysis
#   GPU=0 scripts/run_dory.sh screen       # 50 selection prompts: ablation per site, random, actadd, weight edits
# The screen candidates were chosen from the per-site analysis (analysis/v4_*_sites.csv).
# Dory has no transformers or vLLM release support: loop_steer.dory (HF) and loop_steer.vllm_dory (NVIDIA's vLLM PR,
# ported to vLLM 0.18) are registered in-process. Same sampling as the other models (T=0.6, no top-p / top-k); CoT and
# answer caps 4096; 2 loops as trained. Steps skip what exists. Output goes to the step's log and to stdout.
set -u -o pipefail
cd "$(dirname "$0")/.."
STEP=${1:?step: validate|clean|directions|screen}
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
  screen)  # 50 selection prompts. End-of-prompt separation peaks at recurrent 34-42 (AUC 0.95-0.96, both loops), so
    # ablation spans input (l10), recurrent loop 0 / 1 and output (l62, l70); CoT directions at fewer sites (they broke
    # Ouro and gpt-oss); actadd at two recurrent layers (applied in both loops); weight edits plain and lr-aware (na=1)
    C="none ablate:random:t1.l38"
    for s in l10 t0.l22 t0.l30 t0.l38 t1.l30 t1.l38 t1.l46 t1.l54 l62 l70; do C="$C ablate:v4_baseline:$s"; done
    for s in t0.l22 t1.l38 t1.l54 l62; do C="$C ablate:v4_cot:$s"; done
    for s in t0.l22 t1.l30; do for c in -1 -2; do C="$C actadd:v4_baseline:$s:c=$c"; done; done
    C="$C ortho:v4_baseline:t1.l38 ortho:v4_baseline:t1.l38:na=1"
    # second round (Dory resists: best was t0.l22 at 47%): finer loop-0 grid, per-loop directions, the weight edit at
    # the best hook site, a stronger actadd
    for s in l14 t0.l18 t0.l20 t0.l24 t0.l26 t0.l34; do C="$C ablate:v4_baseline:$s"; done
    C="$C ablate:v4_baseline:t0.l22:dir=perloop ablate:v4_baseline:t0.l38:dir=perloop ortho:v4_baseline:t0.l22"
    C="$C actadd:v4_baseline:t0.l22:c=-1.5"
    # third round: which u works tracks where 0 lies along it (rank corr. -0.95 with the CoT-window class midpoint over
    # sites; -0.11 with AUC), so affine ablation (ACE): set the component to the complying mean instead of 0
    for s in t0.l22 t0.l24 t0.l26 t0.l34 t0.l38 t1.l38; do C="$C ablate:v4_baseline:$s:ace=v4_cot"; done
    C="$C ablate:v4_baseline:t0.l22:ace=v4_baseline ablate:v4_baseline:t1.l38:ace=v4_baseline"
    C="$C ablate:random:t1.l38:ace=v4_cot"
    run scripts/run_select.sh dory select_screen $C ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
