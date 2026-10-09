#!/usr/bin/env bash
# Judge the Nanbeige continuations (the chain's own judge step failed on a missing clean_test_ext/invalid dir).
# Waits for the extension chain to finish so the GPU is free.
cd "$(dirname "$0")/.."
L=data/runs/logs
N=data/runs/Nanbeige4.2-3B/generations
while [ ! -f $L/extend_gpu1.done ]; do sleep 30; done
.venv/bin/python scripts/judge.py --gpu 1 $N/test_int_ext $N/test_int_ext/invalid $N/clean_test_ext \
    > $L/judge_nanbeige_ext.log 2>&1
echo done > $L/judge_nanbeige_ext.done
