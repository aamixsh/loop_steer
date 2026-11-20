#!/bin/bash
# Inkling 2 Mini refusal steering on both GPUs: site selection, then the test-set loop grids (the E1-E3 protocol).
#   scripts/inkling_refusal.sh            (resumable: every step skips existing outputs)
# 1. selection (50 held-out train prompts, offset 500): directional ablation in every unit for 3 methods x 6 sites,
#    plus none and a random direction; GPU 0 / GPU 1 take half each (tags sel_a / sel_b)
# 2. pick the ablation with the lowest all_refusal whose capped share is at most clean + 0.10 -> analysis/inkling_site.json
# 3. GPU 1: ablation grid on the test set (250 prompts): single passes, passes 1+2, all passes, everything (prelude and
#    coda too), random control
#    GPU 0: norm-matched actadd tuning (selection set + MATH-500 100, c = -0.5..-2, every pass), pick c
#    (actadd_tune.py), then the actadd grid on the same test prompts, same units
set -uo pipefail
cd "$(dirname "$0")/.."
source envs/inkling/env.sh
PY=.venv/bin/python; M=thinkingmachines/Inkling-2-Mini; A=data/runs/Inkling-2-Mini/analysis; L=data/runs/logs
SITES="t2.l12 t1.l12 t2.l11 t0.l19 t1.l14 t0.l14"
SEL_A="none ablate:random:t2.l12"; for s in $SITES; do SEL_A="$SEL_A ablate:v4_baseline:$s"; done
SEL_A="$SEL_A ablate:v4_cot:t2.l12 ablate:v4_cot:t1.l12"
SEL_B=""; for s in $SITES; do SEL_B="$SEL_B ablate:hh_eoi:$s"; done
SEL_B="$SEL_B ablate:v4_cot:t2.l11 ablate:v4_cot:t0.l19 ablate:v4_cot:t1.l14 ablate:v4_cot:t0.l14"
# WAIT_PIDS: processes to wait for before selection (e.g. a selection half already running); GPU0_AFTER: a process
# GPU 0 must finish before its tuning starts (e.g. a kick run)
for p in ${WAIT_PIDS:-}; do while kill -0 $p 2>/dev/null; do sleep 15; done; done
if [ -z "${SKIP_SEL:-}" ]; then  # SKIP_SEL=1: selection already judged and summarized (sel_a.csv, sel_b.csv)
  GPU=0 scripts/run_select.sh inkling sel_a $SEL_A > $L/inkling_sel_a.out 2>&1 &
  GPU=1 scripts/run_select.sh inkling sel_b $SEL_B > $L/inkling_sel_b.out 2>&1 &
  wait
fi
$PY - <<PY
import json, pandas as pd
A = "$A"
df = pd.concat([pd.read_csv(f"{A}/sel_a.csv"), pd.read_csv(f"{A}/sel_b.csv")], ignore_index=True)
df.to_csv(f"{A}/sel.csv", index=False)
clean = df[df.candidate == "none"].iloc[0]
abl = df[df.candidate.str.startswith("ablate_") & ~df.candidate.str.contains("random")]
ok = abl[abl.capped_frac <= clean.capped_frac + 0.10].sort_values("all_refusal")
best = ok.iloc[0].candidate
method, site = best.split("_", 1)[1].rsplit("_", 1)  # ablate_<method>_<site>; methods contain "_"
json.dump({"method": method, "site": site, "candidate": best, "all_refusal": float(ok.iloc[0].all_refusal),
           "clean_all_refusal": float(clean.all_refusal)}, open(f"{A}/inkling_site.json", "w"), indent=1)
print(df[["candidate", "all_refusal", "all_score", "capped_frac"]].sort_values("all_refusal").to_string(index=False))
print("picked", method, site)
PY
MS=$($PY -c "import json; d=json.load(open('$A/inkling_site.json')); print(d['method'] + ':' + d['site'])")
SITE=${MS#*:}
SETS=(":apply=0" ":apply=1" ":apply=2" ":apply=0,1" ":apply=0,1,2" "")
GRID_ABL="none ablate:random:$SITE"; for s in "${SETS[@]}"; do GRID_ABL="$GRID_ABL ablate:$MS$s"; done
TEST="SPLIT=test OFFSET=0 NPROMPTS=250"
(env $TEST GPU=1 scripts/run_select.sh inkling grid_test $GRID_ABL) > $L/inkling_grid_abl.out 2>&1 &
(
  for p in ${GPU0_AFTER:-}; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
  TUNE="none"; for c in -0.5 -1 -1.5 -2; do TUNE="$TUNE actadd:$MS:c=$c:rel=1"; done
  GPU=0 scripts/run_select.sh inkling tune_actadd $TUNE
  $PY_INKLING scripts/capability.py --gpu 0 --model $M --dataset math500 --tag tune_actadd --limit 100 \
      --temperature 1.0 --skip-existing --candidates $TUNE
  $PY scripts/actadd_tune.py --model inkling
  COEF=$($PY -c "import json; print(json.load(open('$A/actadd_tune.json'))['c'])")
  GRID_ACT=""; for s in "${SETS[@]}"; do GRID_ACT="$GRID_ACT actadd:$MS:c=$COEF:rel=1$s"; done
  env $TEST GPU=0 scripts/run_select.sh inkling grid_test $GRID_ACT
) > $L/inkling_tune_grid_act.out 2>&1 &
wait
echo "inkling_refusal.sh done"
