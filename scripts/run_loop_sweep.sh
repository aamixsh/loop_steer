#!/usr/bin/env bash
# Does refusal steering survive in different loop iterations of a looped model?
# Ablation (hooks) and activation addition applied in every non-empty subset of the loops (Ouro: 4 loops, 15 subsets;
# Nanbeige: 2 loops, 3 subsets), with the direction of one site used in every selected loop (VARIANT=shared) or with
# each loop's own direction (VARIANT=perloop, `dir=perloop` in scripts/intervene.py). The shared variant also runs a
# clean run and random-direction controls. Same 50 selection prompts as the earlier selection runs (train split,
# offset 500), 3 CoTs per prompt, 3 answers per CoT.
# Skip-existing, so the script can be re-run to resume. Results: data/runs/<model>/generations/$TAG; the summary
# (scripts/loop_sweep_summary.py, run at the end) compares the variants when both exist. Loops are numbered 0-3 in the
# code and 1-4 in the summary.
#   GPU=0 ACTADD_C=-1 scripts/run_loop_sweep.sh                                 # Ouro, shared direction
#   GPU=0 ACTADD_C=-1 VARIANT=perloop scripts/run_loop_sweep.sh                 # Ouro, per-loop directions
#   MODEL_KEY=nanbeige ACTADD_C=-2 VARIANT=shared|perloop scripts/run_loop_sweep.sh
# Other overrides: DIRECTION (method in directions/train.pt), LAYER, SITE_LOOP, TAG. DRY=1 lists the candidates.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=data/runs/logs
G=${GPU:-0}
MODEL_KEY=${MODEL_KEY:-ouro}
VARIANT=${VARIANT:-shared}
DIRECTION=${DIRECTION:-v4_baseline}   # direction method (directions/train.pt)
ACTADD_C=${ACTADD_C:--1}              # actadd coefficient on the raw mean-difference vector (negative = subtract)
case $MODEL_KEY in
  ouro)     MODEL=ByteDance/Ouro-1.4B-Thinking; RUN=Ouro-1.4B-Thinking; N_LOOPS=4; SITE_LOOP=${SITE_LOOP:-3}
            LAYER=${LAYER:-16}; GEN_ARGS=""; MAX_TOKENS=2048 ;;
  nanbeige) MODEL=Nanbeige/Nanbeige4.2-3B; RUN=Nanbeige4.2-3B; N_LOOPS=2; SITE_LOOP=${SITE_LOOP:-1}
            LAYER=${LAYER:-15}; GEN_ARGS="--max-tokens 4096 --max-model-len 12288"; MAX_TOKENS=4096 ;;
  *) echo "MODEL_KEY must be ouro or nanbeige" >&2; exit 2 ;;
esac
case $VARIANT in
  shared)  TAG=${TAG:-loop_sweep}; DIRSFX="" ;;
  perloop) TAG=${TAG:-loop_sweep_perloop}; DIRSFX=":dir=perloop" ;;
  *) echo "VARIANT must be shared or perloop" >&2; exit 2 ;;
esac
OUT=data/runs/$RUN/generations/$TAG
SITE=t$SITE_LOOP.l$LAYER

# All non-empty loop subsets, smallest first. The full set is the default (no apply=).
SUBSETS=$($PY -I -c "
from itertools import combinations
n = $N_LOOPS
for k in range(1, n + 1):
    for s in combinations(range(n), k):
        print('all' if k == n else ':apply=' + ','.join(map(str, s)))
")
if [ "$VARIANT" = shared ]; then
  # Matched-norm random control for actadd: scale the coefficient by |refusal vector| / |random vector| at the site.
  RAND_C=$($PY -I - <<EOF
import torch
d = torch.load("data/runs/$RUN/directions/train.pt", map_location="cpu", weights_only=False)
i = d["sites"].index(($SITE_LOOP, $LAYER))
print(f"{$ACTADD_C * d['$DIRECTION']['dirs'][i].float().norm().item() / d['random']['dirs'][i].float().norm().item():.4f}")
EOF
)
  CANDS=(none "ablate:random:$SITE" "actadd:random:$SITE:c=$RAND_C")
else
  RAND_C=n/a
  CANDS=("ablate:random:$SITE:dir=perloop")   # null control: a random direction per loop
fi
for kind in ablate actadd; do
  while IFS= read -r sub; do
    [ "$sub" = all ] && sub=""
    if [ "$kind" = actadd ]; then CANDS+=("actadd:$DIRECTION:$SITE:c=$ACTADD_C$sub$DIRSFX"); else CANDS+=("ablate:$DIRECTION:$SITE$sub$DIRSFX"); fi
  done <<< "$SUBSETS"
done
echo "$MODEL_KEY $VARIANT: ${#CANDS[@]} candidates; actadd c=$ACTADD_C (random control c=$RAND_C); output $OUT"
if [ -n "${DRY:-}" ]; then printf '%s\n' "${CANDS[@]}"; exit 0; fi   # DRY=1: list the candidates and stop

$PY scripts/intervene.py --gpu $G --model $MODEL --directions train --split train --offset 500 --n-prompts 50 $GEN_ARGS \
    --tag $TAG --skip-existing --candidates "${CANDS[@]}" > $L/${MODEL_KEY}_$TAG.log 2>&1
$PY scripts/judge.py --gpu $G $OUT $OUT/invalid > $L/judge_${MODEL_KEY}_$TAG.log 2>&1
# compare with the shared-direction run when this is the per-loop variant and that one exists
TAGS=$TAG; [ "$VARIANT" = perloop ] && [ -d data/runs/$RUN/generations/loop_sweep ] && TAGS="loop_sweep $TAG"
$PY scripts/loop_sweep_summary.py $TAGS --model $RUN --n-loops $N_LOOPS --max-tokens $MAX_TOKENS \
    > $L/summary_${MODEL_KEY}_$TAG.log 2>&1
echo done > $L/${MODEL_KEY}_$TAG.done
