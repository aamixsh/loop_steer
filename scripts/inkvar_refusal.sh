#!/bin/bash
# Refusal steering on TML-d24-varloop-research (Inkling's variable-pass sibling; trained with 1-5 passes, post-trained
# at 3) on both GPUs, after scripts/inkling_directions.sh (MODEL=...varloop...) has written its directions:
# 1. selection (50 held-out train prompts): ablation in every unit, 3 methods x 5 sites, plus none and random;
#    pick the lowest all_refusal with capped share <= clean + 0.10 -> analysis/inkling_site.json
# 2. GPU 1: ablation grids on the test set (250 prompts): at 3 passes (single passes, 1+2, all passes, everything,
#    random) and at 5 passes ("add laps": each single pass, the trained passes 1-3, all five, everything)
#    GPU 0: stream norms at 5 passes, actadd tuning at 3 passes (selection set + MATH-500 100), pick c, then the actadd
#    grids at 3 and 5 passes (same units)
# Resumable (every step skips existing outputs).
set -uo pipefail
cd "$(dirname "$0")/.."
source envs/inkling/env.sh
PY=.venv/bin/python; K=inkvar; M=thinkingmachines/TML-d24-varloop-research
RUN=data/runs/TML-d24-varloop-research; A=$RUN/analysis; L=data/runs/logs
until [ -f $RUN/directions/train.pt ] && [ -f $RUN/directions/stream_norms.pt ]; do sleep 30; done
for p in ${WAIT_PIDS:-}; do while kill -0 $p 2>/dev/null; do sleep 15; done; done
SITES="t2.l11 t2.l12 t1.l11 t0.l14 t1.l14"
SEL_A="none ablate:random:t2.l11"; for s in $SITES; do SEL_A="$SEL_A ablate:v4_cot:$s"; done
SEL_A="$SEL_A ablate:v4_baseline:t2.l11 ablate:v4_baseline:t1.l11"
SEL_B=""; for s in $SITES; do SEL_B="$SEL_B ablate:hh_eoi:$s"; done
SEL_B="$SEL_B ablate:v4_baseline:t2.l12 ablate:v4_baseline:t0.l14 ablate:v4_baseline:t1.l14"
if [ ! -f $A/inkling_site.json ]; then
  GPU=0 scripts/run_select.sh $K sel_a $SEL_A > $L/inkvar_sel_a.out 2>&1 &
  GPU=1 scripts/run_select.sh $K sel_b $SEL_B > $L/inkvar_sel_b.out 2>&1 &
  wait
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
fi
MS=$($PY -c "import json; d=json.load(open('$A/inkling_site.json')); print(d['method'] + ':' + d['site'])")
SITE=${MS#*:}
SETS3=(":apply=0" ":apply=1" ":apply=2" ":apply=0,1" ":apply=0,1,2" "")
SETS5=(":apply=0" ":apply=1" ":apply=2" ":apply=3" ":apply=4" ":apply=0,1,2" ":apply=0,1,2,3,4" "")
TEST="SPLIT=test OFFSET=0 NPROMPTS=250"
(
  G3="none ablate:random:$SITE"; for s in "${SETS3[@]}"; do G3="$G3 ablate:$MS$s"; done
  env $TEST GPU=1 scripts/run_select.sh $K grid_test $G3
  G5="none"; for s in "${SETS5[@]}"; do G5="$G5 ablate:$MS$s"; done
  env $TEST N_LOOPS=5 GPU=1 scripts/run_select.sh $K grid_test5 $G5
) > $L/inkvar_grid_abl.out 2>&1 &
(
  $PY_INKLING scripts/stream_norms.py --gpu 0 --model $M --n-loops 5 --skip-existing
  TUNE="none"; for c in -0.5 -1 -1.5 -2; do TUNE="$TUNE actadd:$MS:c=$c:rel=1"; done
  GPU=0 scripts/run_select.sh $K tune_actadd $TUNE
  $PY_INKLING scripts/capability.py --gpu 0 --model $M --dataset math500 --tag tune_actadd --limit 100 \
      --temperature 1.0 --skip-existing --candidates $TUNE
  $PY scripts/actadd_tune.py --model $K
  COEF=$($PY -c "import json; print(json.load(open('$A/actadd_tune.json'))['c'])")
  G3=""; for s in "${SETS3[@]}"; do G3="$G3 actadd:$MS:c=$COEF:rel=1$s"; done
  env $TEST GPU=0 scripts/run_select.sh $K grid_test_act $G3  # own tags: the two GPUs never judge one dir
  G5=""; for s in "${SETS5[@]}"; do G5="$G5 actadd:$MS:c=$COEF:rel=1$s"; done
  env $TEST N_LOOPS=5 GPU=0 scripts/run_select.sh $K grid_test5_act $G5
) > $L/inkvar_tune_grid_act.out 2>&1 &
wait
echo "inkvar_refusal.sh done"
