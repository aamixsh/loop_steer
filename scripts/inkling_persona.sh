#!/bin/bash
# Persona vectors on Inkling 2 Mini (the E5 / E6 / E11 protocol), one GPU ($GPU): contrastive generations, judge,
# vectors (vLLM capture, NVFP4 as served), strength x site tuning (every pass), steering in single passes / passes 1+2
# / every pass, and ablation of system-prompt-elicited traits in the same units. Generation runs in .venv-inkling,
# judging in .venv (gpt-oss-20b). Every stage skips existing outputs; rerun to resume.
#   GPU=1 scripts/inkling_persona.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source envs/inkling/env.sh
M=thinkingmachines/Inkling-2-Mini; G=${GPU:?set GPU}; PY=.venv/bin/python
SETS="1=0 2=1 3=2 12=0,1 all=0,1,2"
$PY_INKLING scripts/persona_vectors.py --gpu $G --model $M --stage gen
$PY scripts/persona_vectors.py --gpu $G --model $M --stage judge --file extract_gen
$PY_INKLING scripts/persona_vectors.py --gpu $G --model $M --stage extract
$PY_INKLING scripts/persona_steer.py --gpu $G --model $M --mode tune --stages steer
$PY scripts/persona_steer.py --gpu $G --model $M --mode tune --stages judge,summary
$PY_INKLING scripts/persona_steer.py --gpu $G --model $M --mode loops --sets $SETS --stages steer
$PY scripts/persona_steer.py --gpu $G --model $M --mode loops --sets $SETS --stages judge,summary
$PY_INKLING scripts/persona_steer.py --gpu $G --model $M --mode ablate --sets $SETS --stages steer
$PY scripts/persona_steer.py --gpu $G --model $M --mode ablate --sets $SETS --stages judge,summary
echo "inkling_persona.sh done"
