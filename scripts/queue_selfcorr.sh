#!/usr/bin/env bash
# Self-correction experiments (docs/paper_plan.md, section C), queued on both GPUs with scripts/gpuq.py.
#   GPU 1: impulse response + contraction, loop-boundary patching, dense-control rebuild, extra loops, CAA, CAA impulses
#   GPU 0: neuron pushes, persona vectors and persona impulses for Ouro and Nanbeige (Dory and Qwen: GPU 1, after CAA)
# Every step is skip-existing, so re-running this script after a failure only redoes what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OURO=ByteDance/Ouro-1.4B-Thinking; NANB=Nanbeige/Nanbeige4.2-3B; DORY=nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
QWEN=Qwen/Qwen3-8B; GPTOSS=openai/gpt-oss-20b
q() { local gpu=$1 name=$2; shift 2; scripts/gpuq.py add -g "$gpu" -n "$name" --idle 3600 --retries 1 -- "$*"; }
key() { case $1 in "$OURO") echo ouro;; "$NANB") echo nanbeige;; "$DORY") echo dory;; "$QWEN") echo qwen;; "$GPTOSS") echo gptoss;; esac; }
mid() { case $1 in "$OURO") echo t3.l12;; "$NANB") echo t1.l11;; "$DORY") echo t1.l36;; "$QWEN") echo l18;; esac; }
CAA="coordinate-other-ais corrigible-neutral-HHH hallucination myopic-reward survival-instinct sycophancy refusal"
TRAITS="evil sycophantic hallucinating humorous optimistic impolite apathetic"

# ---------------- GPU 1
for m in $OURO $NANB $DORY $QWEN $GPTOSS; do
  q 1 "perturb-$(key $m)" "$PY scripts/perturb_response.py --gpu \$GPU --model $m --skip-existing"
done
for m in $OURO $NANB $DORY $QWEN $GPTOSS; do
  q 1 "patch-$(key $m)" "$PY scripts/loop_patch.py --gpu \$GPU --model $m --skip-existing"
done
q 1 rebuild-qwen-blocks "$PY scripts/loop_rebuild.py --gpu \$GPU --model $QWEN --direction v4_baseline:l21 --blocks 4"
q 1 rebuild-gptoss-blocks "$PY scripts/loop_rebuild.py --gpu \$GPU --model $GPTOSS --direction v4_baseline:l15 --blocks 4 \
  --gen-path train_clean.scored.parquet"
q 1 rebuild-dory "$PY scripts/loop_rebuild.py --gpu \$GPU --model $DORY --direction hh_eoi:t0.l26 \
  --gen-path train_clean.scored.parquet"
q 1 perturb-ouro-loops8 "$PY scripts/perturb_response.py --gpu \$GPU --model $OURO --n-loops 8 --tag loops8 --skip-existing"
q 1 perturb-nanbeige-loops4 "$PY scripts/perturb_response.py --gpu \$GPU --model $NANB --n-loops 4 --tag loops4 --skip-existing"
q 1 perturb-dory-loops6 "$PY scripts/perturb_response.py --gpu \$GPU --model $DORY --n-loops 6 --tag loops6 --skip-existing"
for m in $OURO $NANB $DORY $QWEN; do
  q 1 "caa-$(key $m)" "$PY scripts/caa_loops.py --gpu \$GPU --model $m"
done
for m in $OURO $NANB $DORY $QWEN; do
  vecs=""; for b in $CAA; do vecs+=" --vec caa_$b=caa:$b:$(mid $m)"; done
  q 1 "perturb-caa-$(key $m)" "$PY scripts/perturb_response.py --gpu \$GPU --model $m --tag caa --n-random 2 $vecs --skip-existing"
done

# ---------------- GPU 0 (persona work for Dory and Qwen goes to GPU 1, which finishes its own jobs first)
for m in $OURO $NANB $DORY $QWEN; do
  NP="$PY scripts/neuron_push.py --gpu \$GPU --model $m"
  q 0 "neurons-$(key $m)" "$NP --stage gen && $NP --stage judge && $NP --stage summary"
done
persona() {  # gpu, model: persona-vector chain
  local PV="$PY scripts/persona_vectors.py --gpu \$GPU --model $2"
  q "$1" "persona-$(key $2)" "$PV --stage gen && $PV --stage judge --file extract_gen && $PV --stage extract && \
$PV --stage steer && $PV --stage judge --file steer_gen && $PV --stage summary"
}
persona_impulse() {  # gpu, model: impulse response with the persona vectors (after persona())
  local vecs=""; for t in $TRAITS; do vecs+=" --vec persona_$t=persona:$t:$(mid $2)"; done
  q "$1" "perturb-persona-$(key $2)" "$PY scripts/perturb_response.py --gpu \$GPU --model $2 --tag persona --n-random 2 $vecs --skip-existing"
}
persona 0 $OURO; persona 0 $NANB; persona_impulse 0 $OURO; persona_impulse 0 $NANB
persona 1 $DORY; persona 1 $QWEN; persona_impulse 1 $DORY; persona_impulse 1 $QWEN
scripts/gpuq.py list -g 0; scripts/gpuq.py list -g 1
