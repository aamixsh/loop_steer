#!/usr/bin/env bash
# Full test-set (487 prompts) interventions on Nanbeige4.2-3B, GPU 1, then judging.
# Skip-existing, so the script can be re-run to resume. Logs go to data/runs/logs/.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=data/runs/logs
N=data/runs/Nanbeige4.2-3B/generations

$PY scripts/intervene.py --gpu 1 --model Nanbeige/Nanbeige4.2-3B --directions train --split test \
    --max-tokens 4096 --max-model-len 12288 --tag test_int --skip-existing \
    --candidates ablate:v4_baseline:t1.l15 ortho:v4_baseline:t1.l15 ablate:random:t0.l14 \
                 ortho:random:t0.l14 ablate:v4_cot:t0.l14 \
    > $L/nanbeige_test_int.log 2>&1
$PY scripts/judge.py --gpu 1 $N/test_int $N/test_int/invalid > $L/judge_nanbeige_test_int.log 2>&1
