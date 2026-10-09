#!/usr/bin/env bash
# Redo the CoTs that hit the token cap, on GPU 1. Skip-existing, so the script can be re-run to resume.
#  1. Continue capped CoTs of the test-set runs (same intervention) up to 8192 tokens in total, answer them, judge.
#  2. Regenerate the clean Qwen / Ouro test runs with an 8192-token CoT cap (the old ones did not save the
#     capped CoTs), keeping unclosed ones in invalid/.
# Continuations go to <tag>_ext; summarize.py --cont merges them. Logs go to data/runs/logs/.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=data/runs/logs
G=${GPU:-1}  # GPU index (override: GPU=0 scripts/run_extend_gpu1.sh)
Q=data/runs/Qwen3-8B/generations
O=data/runs/Ouro-1.4B-Thinking/generations
N=data/runs/Nanbeige4.2-3B/generations
LEN="--max-model-len 16384"

# --- Qwen3-8B (cap was 2048) ---
$PY scripts/intervene.py --gpu $G --model Qwen/Qwen3-8B --directions train --split test --tag test_ortho_ext \
    --extend-from test_ortho --max-tokens 2048 --extend-tokens 6144 $LEN --skip-existing \
    --candidates ortho:v4_baseline:l21 ortho:v4_baseline:l17 ortho:v12_cot150:l17 ortho:v4_cot:l15 \
                 ortho:v4_cot:l23 ortho:random:l17 > $L/qwen_ext.log 2>&1
$PY scripts/judge.py --gpu $G $Q/test_ortho_ext $Q/test_ortho_ext/invalid > $L/judge_qwen_ext.log 2>&1

# --- Ouro (cap was 2048) ---
$PY scripts/intervene.py --gpu $G --model ByteDance/Ouro-1.4B-Thinking --directions train --split test \
    --tag test_hooks_ext --extend-from test_hooks --max-tokens 2048 --extend-tokens 6144 $LEN --skip-existing \
    --candidates ablate:v4_baseline:t3.l16 ablate:random:t3.l16 ablate:v4_baseline:t3.l8 > $L/ouro_ext.log 2>&1
$PY scripts/judge.py --gpu $G $O/test_hooks_ext $O/test_hooks_ext/invalid > $L/judge_ouro_ext.log 2>&1

# --- Nanbeige (cap was 4096): interventions, then the clean run ---
$PY scripts/intervene.py --gpu $G --model Nanbeige/Nanbeige4.2-3B --directions train --split test \
    --tag test_int_ext --extend-from test_int --max-tokens 4096 --extend-tokens 4096 $LEN --skip-existing \
    --candidates ablate:v4_baseline:t1.l15 ortho:v4_baseline:t1.l15 ablate:random:t0.l14 \
                 ortho:random:t0.l14 ablate:v4_cot:t0.l14 > $L/nanbeige_ext.log 2>&1
$PY scripts/intervene.py --gpu $G --model Nanbeige/Nanbeige4.2-3B --split test \
    --tag clean_test_ext --extend-from clean_test --max-tokens 4096 --extend-tokens 4096 $LEN --skip-existing \
    --candidates none > $L/nanbeige_clean_ext.log 2>&1
$PY scripts/judge.py --gpu $G $N/test_int_ext $N/test_int_ext/invalid $N/clean_test_ext $N/clean_test_ext/invalid \
    > $L/judge_nanbeige_ext.log 2>&1

# --- Clean Qwen / Ouro with an 8192 CoT cap (answers keep the old 2048 cap) ---
$PY scripts/intervene.py --gpu $G --model Qwen/Qwen3-8B --split test --tag clean_test_8k --max-tokens 8192 \
    --answer-max-tokens 2048 --max-model-len 16384 --skip-existing --candidates none > $L/qwen_clean_8k.log 2>&1
$PY scripts/intervene.py --gpu $G --model ByteDance/Ouro-1.4B-Thinking --split test --tag clean_test_8k \
    --max-tokens 8192 --answer-max-tokens 2048 --max-model-len 16384 --skip-existing --candidates none \
    > $L/ouro_clean_8k.log 2>&1
$PY scripts/judge.py --gpu $G $Q/clean_test_8k $Q/clean_test_8k/invalid $O/clean_test_8k $O/clean_test_8k/invalid \
    > $L/judge_clean_8k.log 2>&1
echo done > $L/extend_gpu1.done
