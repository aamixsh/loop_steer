#!/usr/bin/env bash
# Full test-set (487 prompts) interventions, GPU 0: Qwen remaining ortho candidates, then Ouro hooks.
# Each stage is skip-existing, so the script can be re-run to resume. Logs go to data/runs/logs/.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
GPU=${GPU:-0}  # GPU index (override: GPU=1 scripts/...)
L=data/runs/logs
Q=data/runs/Qwen3-8B/generations
O=data/runs/Ouro-1.4B-Thinking/generations

$PY scripts/intervene.py --gpu $GPU --model Qwen/Qwen3-8B --directions train --split test --tag test_ortho \
    --skip-existing --candidates ortho:v4_baseline:l17 ortho:v4_baseline:l21 ortho:random:l17 \
    > $L/qwen_test_ortho_part2.log 2>&1
$PY scripts/judge.py --gpu $GPU $Q/test_ortho $Q/test_ortho/invalid > $L/judge_qwen_test_ortho_part2.log 2>&1

$PY scripts/intervene.py --gpu $GPU --gpu-memory-utilization 0.5 --model ByteDance/Ouro-1.4B-Thinking \
    --directions train --split test --tag test_hooks --skip-existing \
    --candidates ablate:v4_baseline:t3.l16 ablate:random:t3.l16 ablate:v4_baseline:t3.l8 \
    > $L/ouro_test_hooks.log 2>&1
$PY scripts/judge.py --gpu $GPU $O/test_hooks $O/test_hooks/invalid > $L/judge_ouro_test_hooks.log 2>&1
