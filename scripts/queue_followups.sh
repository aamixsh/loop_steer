#!/usr/bin/env bash
# Waits for the per-loop-direction chain (scripts/queue_perloop_chain.sh) to finish, then runs the follow-up
# experiments in order on one GPU. Steps are defined in scripts/run_followups.sh (re-read for every step).
cd "$(dirname "$0")/.."
L=data/runs/logs
while [ ! -f $L/perloop_chain.done ]; do sleep 60; done
GPU=${GPU:-0} scripts/run_followups.sh gates overrefusal confirm layermap normaware qwentest capability
echo done > $L/followups.done
