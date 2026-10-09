#!/usr/bin/env bash
# Wait for another job (PID in $1) to exit, then run the pending Qwen3-8B test-set ortho candidates
# on GPU 0 and judge them. Skip-existing, so it can be re-run to resume.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
PY=.venv/bin/python
GPU=${GPU:-0}  # GPU index (override: GPU=1 scripts/...)
L=data/runs/logs
Q=data/runs/Qwen3-8B/generations

$PY scripts/intervene.py --gpu $GPU --model Qwen/Qwen3-8B --directions train --split test --tag test_ortho \
    --skip-existing --candidates ortho:v4_baseline:l17 ortho:v4_baseline:l21 ortho:random:l17 \
    > $L/qwen_test_ortho_part2.log 2>&1
$PY scripts/judge.py --gpu $GPU $Q/test_ortho $Q/test_ortho/invalid > $L/judge_qwen_test_ortho_part2.log 2>&1
