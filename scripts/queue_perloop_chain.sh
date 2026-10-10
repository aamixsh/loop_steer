#!/usr/bin/env bash
# One GPU-0 queue: (1) Nanbeige actadd calibration, (2) Ouro per-loop-direction sweep, (3) Nanbeige sweeps (shared and
# per-loop directions) with the actadd coefficient read from data/runs/logs/nanbeige_actadd_c (default -2).
# Each step is skip-existing, so the queue can be re-run to resume.   GPU=0 scripts/queue_perloop_chain.sh
cd "$(dirname "$0")/.."
G=${GPU:-0}
L=data/runs/logs
N=data/runs/Nanbeige4.2-3B/generations/loop_sweep_pilot

# 1. actadd strength on Nanbeige: all loops, layer 15, the 50 selection prompts (c=-1 exists in generations/sel)
.venv/bin/python scripts/intervene.py --gpu $G --model Nanbeige/Nanbeige4.2-3B --directions train --split train \
    --offset 500 --n-prompts 50 --max-tokens 4096 --max-model-len 12288 --tag loop_sweep_pilot --skip-existing \
    --candidates actadd:v4_baseline:t1.l15:c=-2 actadd:v4_baseline:t1.l15:c=-3 actadd:v4_baseline:t1.l15:c=-4 \
    > $L/nanbeige_loop_pilot.log 2>&1
.venv/bin/python scripts/judge.py --gpu $G $N $N/invalid > $L/judge_nanbeige_loop_pilot.log 2>&1
echo done > $L/nanbeige_loop_pilot.done

# 2. Ouro, per-loop directions (the shared-direction sweep is loop_sweep; this one is compared against it)
GPU=$G MODEL_KEY=ouro VARIANT=perloop ACTADD_C=-1 scripts/run_loop_sweep.sh

# 3. Nanbeige
C=$(cat $L/nanbeige_actadd_c 2>/dev/null || echo -2)
GPU=$G MODEL_KEY=nanbeige VARIANT=shared ACTADD_C=$C scripts/run_loop_sweep.sh
GPU=$G MODEL_KEY=nanbeige VARIANT=perloop ACTADD_C=$C scripts/run_loop_sweep.sh
echo done > $L/perloop_chain.done
