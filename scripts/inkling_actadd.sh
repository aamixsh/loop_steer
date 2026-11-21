#!/bin/bash
# Inkling 2 Mini norm-matched activation addition (the E2 + E3 protocol) for the direction chosen by the selection
# (analysis/inkling_site.json): tune c on the selection set + MATH-500 (100), pick it (actadd_tune.py), then the test
# grid (250 prompts) in single passes, passes 1+2, all passes, and everything. GPU from $GPU. Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
source envs/inkling/env.sh
PY=.venv/bin/python; M=thinkingmachines/Inkling-2-Mini; A=data/runs/Inkling-2-Mini/analysis
MS=$($PY -c "import json; d=json.load(open('$A/inkling_site.json')); print(d['method'] + ':' + d['site'])")
TUNE="none"; for c in -0.5 -1 -1.5 -2; do TUNE="$TUNE actadd:$MS:c=$c:rel=1"; done
GPU=$GPU scripts/run_select.sh inkling tune_actadd $TUNE
$PY_INKLING scripts/capability.py --gpu $GPU --model $M --dataset math500 --tag tune_actadd --limit 100 \
    --temperature 1.0 --skip-existing --candidates $TUNE
$PY scripts/actadd_tune.py --model inkling
COEF=$($PY -c "import json; print(json.load(open('$A/actadd_tune.json'))['c'])")
GRID=""; for s in ":apply=0" ":apply=1" ":apply=2" ":apply=0,1" ":apply=0,1,2" ""; do GRID="$GRID actadd:$MS:c=$COEF:rel=1$s"; done
SPLIT=test OFFSET=0 NPROMPTS=250 GPU=$GPU scripts/run_select.sh inkling grid_test $GRID
