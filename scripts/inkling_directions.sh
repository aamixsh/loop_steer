#!/bin/bash
# Inkling 2 Mini refusal directions from the two train halves (generate.py --offset 0 / 250, 250 prompts each):
# judge each half on its own GPU, merge, labels, activations at every site (vLLM, NVFP4), directions incl. hh_eoi.
set -euo pipefail
cd "$(dirname "$0")/.."
source envs/inkling/env.sh
M=thinkingmachines/Inkling-2-Mini
R=data/runs/Inkling-2-Mini/generations
until [ -f $R/train_clean_a.parquet ] && [ -f $R/train_clean_b.parquet ]; do sleep 20; done
while pgrep -f "scripts/generate.py --gpu [01] --model $M" >/dev/null; do sleep 10; done  # GPU memory freed
.venv/bin/python scripts/judge.py --gpu 0 --gpu-memory-utilization 0.85 $R/train_clean_a.parquet &
.venv/bin/python scripts/judge.py --gpu 1 --gpu-memory-utilization 0.85 $R/train_clean_b.parquet &
wait
.venv/bin/python - <<PY
import pandas as pd
R = "$R"
df = pd.concat([pd.read_parquet(f"{R}/train_clean_{h}.scored.parquet") for h in "ab"], ignore_index=True)
df.to_parquet(f"{R}/train_clean.scored.parquet")
print(f"merged {len(df)} rows, {df.prompt_idx.nunique()} prompts")
PY
.venv/bin/python scripts/build_sets.py $R/train_clean.scored.parquet
$PY_INKLING scripts/extract_activations.py --gpu 0 --model $M --labels $R/labels.parquet --name train --labeled-only
.venv/bin/python scripts/directions.py --model $M --activations train --contrast contrast
