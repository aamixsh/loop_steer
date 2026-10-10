#!/usr/bin/env bash
# Follow-up experiments on the looped models, one function per step (run in the order given on the command line):
#   gates        tiny smoke runs of the code paths the later steps rely on (capability.py, norm-aware ortho, harmless
#                prompts + judge); a failure writes data/runs/logs/gate_<name>.fail and the dependent step is skipped
#   overrefusal  harmless test prompts under the main interventions (does restricting loops cut over-refusal?)
#   confirm      test-set runs of the loop configs picked from the selection-set sweeps (Ouro), capped CoTs continued
#   layermap     Ouro single-loop x single-layer actadd map (selection prompts, per-loop directions)
#   normaware    Ouro weight orthogonalization with and without the norm-aware edit (selection prompts)
#   qwentest     Qwen3-8B hook ablation and actadd on the test set (only weight edits were run there before)
#   capability   MATH-500 and AIME 2025 accuracy under the main interventions (Ouro, Nanbeige)
# Everything is skip-existing, so re-running resumes. Logs: data/runs/logs/. Uses GPU $GPU (default 0).
#   scripts/run_followups.sh gates overrefusal confirm layermap normaware qwentest capability
# Nanbeige actadd coefficient: data/runs/logs/nanbeige_actadd_c (default -2). Test-set configs for `confirm`:
# data/runs/logs/confirm_candidates_ouro.txt (space-separated intervene.py candidates; default below).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=data/runs/logs
G=${GPU:-0}
OURO=ByteDance/Ouro-1.4B-Thinking; NB=Nanbeige/Nanbeige4.2-3B; QW=Qwen/Qwen3-8B
O=data/runs/Ouro-1.4B-Thinking; N=data/runs/Nanbeige4.2-3B; Q=data/runs/Qwen3-8B
NB_C=$(cat $L/nanbeige_actadd_c 2>/dev/null || echo -2)
NB_ARGS="--max-tokens 4096 --max-model-len 12288"
SEL="--directions train --split train --offset 500 --n-prompts 50"
mkdir -p $O/analysis $N/analysis $Q/analysis

failed() { [ -f "$L/gate_$1.fail" ] && { echo "skipping: gate $1 failed (see $L/gate_$1.log)"; return 0; }; return 1; }
judge() { $PY scripts/judge.py --gpu $G "$@"; }
intervene() { $PY scripts/intervene.py --gpu $G "$@"; }

step_gates() {
  rm -f $L/gate_*.fail
  $PY scripts/capability.py --gpu $G --model $OURO --dataset math500 --limit 4 --max-tokens 256 --max-model-len 2048 \
      --tag gate --candidates none ablate:v4_baseline:t3.l16:apply=2 > $L/gate_capability.log 2>&1 || touch $L/gate_capability.fail
  [ -f $O/capability/math500/gate/ablate_v4_baseline_t3.l16_apply2.parquet ] || touch $L/gate_capability.fail
  rm -rf $O/capability/math500/gate
  intervene --model $OURO --directions train --split train --offset 500 --n-prompts 2 --cot-reps 1 --out-reps 1 \
      --max-tokens 256 --max-model-len 2048 --tag gate_na \
      --candidates ortho:v4_baseline:t3.l16:na=1 none ablate:v4_baseline:t3.l16:apply=0:dir=perloop \
      > $L/gate_normaware.log 2>&1 || touch $L/gate_normaware.fail
  [ -f $O/generations/gate_na/none.parquet ] || touch $L/gate_normaware.fail
  rm -rf $O/generations/gate_na
  intervene --model $OURO --directions train --split test --kind harmless --n-prompts 6 --cot-reps 1 --out-reps 1 \
      --max-tokens 1024 --max-model-len 4096 --tag gate_harmless --candidates none > $L/gate_harmless.log 2>&1 \
      && judge $O/generations/gate_harmless $O/generations/gate_harmless/invalid >> $L/gate_harmless.log 2>&1 \
      || touch $L/gate_harmless.fail
  rm -rf $O/generations/gate_harmless
  ls $L/gate_*.fail 2>/dev/null || echo "all gates passed"
}

step_overrefusal() {  # harmless test prompts (250): 1 CoT, 2 answers each
  failed harmless && return
  H="--directions train --split test --kind harmless --cot-reps 1 --out-reps 2 --tag overrefusal --skip-existing"
  intervene --model $OURO $H --candidates none ablate:random:t3.l16 ablate:v4_baseline:t3.l16 \
      ablate:v4_baseline:t3.l16:apply=2 ablate:v4_baseline:t3.l16:apply=0 ablate:v4_baseline:t3.l16:apply=0,1 \
      actadd:v4_baseline:t3.l16:c=-1 > $L/ouro_overrefusal.log 2>&1
  intervene --model $NB $H $NB_ARGS --candidates none ablate:random:t1.l15 ablate:v4_baseline:t1.l15 \
      ablate:v4_baseline:t1.l15:apply=0 ablate:v4_baseline:t1.l15:apply=1 actadd:v4_baseline:t1.l15:c=$NB_C \
      > $L/nanbeige_overrefusal.log 2>&1
  judge $O/generations/overrefusal $O/generations/overrefusal/invalid $N/generations/overrefusal \
      $N/generations/overrefusal/invalid > $L/judge_overrefusal.log 2>&1
  $PY scripts/summarize.py $O/generations/overrefusal --csv $O/analysis/overrefusal.csv > $L/summary_ouro_overrefusal.log 2>&1
  $PY scripts/summarize.py $N/generations/overrefusal --max-tokens 4096 --csv $N/analysis/overrefusal.csv \
      > $L/summary_nanbeige_overrefusal.log 2>&1
}

step_confirm() {  # Ouro, all 487 test prompts; capped CoTs continued to 8192 tokens in total
  CANDS=$(cat $L/confirm_candidates_ouro.txt 2>/dev/null || echo "ablate:v4_baseline:t3.l16:apply=2 ablate:v4_baseline:t3.l16:apply=0 ablate:v4_baseline:t3.l16:apply=0,1 actadd:v4_baseline:t3.l16:c=-1")
  echo "confirm candidates: $CANDS"
  intervene --model $OURO --directions train --split test --tag test_loops --skip-existing --candidates $CANDS \
      > $L/ouro_test_loops.log 2>&1
  judge $O/generations/test_loops $O/generations/test_loops/invalid > $L/judge_ouro_test_loops.log 2>&1
  intervene --model $OURO --directions train --split test --tag test_loops_ext --extend-from test_loops \
      --max-tokens 2048 --extend-tokens 6144 --max-model-len 16384 --skip-existing --candidates $CANDS \
      > $L/ouro_test_loops_ext.log 2>&1
  judge $O/generations/test_loops_ext $O/generations/test_loops_ext/invalid > $L/judge_ouro_test_loops_ext.log 2>&1
  $PY scripts/summarize.py $O/generations/test_loops --cont $O/generations/test_loops_ext \
      --csv $O/analysis/test_loops.csv > $L/summary_ouro_test_loops.log 2>&1
}

step_layermap() {  # actadd c=-2 in one loop at one layer, with that loop's own direction at that layer
  CANDS=""
  for t in 0 1 2 3; do for l in 4 8 12 16 20; do CANDS="$CANDS actadd:v4_baseline:t3.l$l:c=-2:apply=$t:dir=perloop"; done; done
  intervene --model $OURO $SEL --tag loop_layer_map --skip-existing --candidates $CANDS > $L/ouro_loop_layer_map.log 2>&1
  judge $O/generations/loop_layer_map $O/generations/loop_layer_map/invalid > $L/judge_ouro_loop_layer_map.log 2>&1
  $PY scripts/loop_layer_summary.py loop_layer_map > $L/summary_ouro_loop_layer_map.log 2>&1
}

step_normaware() {  # weight edits (all loops at once) with and without the norm-aware projection, vs hook ablation
  failed normaware && return
  intervene --model $OURO $SEL --tag ortho_na --skip-existing --candidates ortho:v4_baseline:t3.l16 \
      ortho:v4_baseline:t3.l16:na=1 ortho:v4_cot:t3.l16:na=1 ortho:random:t3.l16:na=1 > $L/ouro_ortho_na.log 2>&1
  judge $O/generations/ortho_na $O/generations/ortho_na/invalid > $L/judge_ouro_ortho_na.log 2>&1
  $PY scripts/summarize.py $O/generations/ortho_na --csv $O/analysis/ortho_na.csv > $L/summary_ouro_ortho_na.log 2>&1
}

step_qwentest() {
  CANDS="ablate:v4_baseline:l21 actadd:v4_baseline:l17:c=-2 ablate:random:l17"
  intervene --model $QW --directions train --split test --tag test_hooks --skip-existing --candidates $CANDS \
      > $L/qwen_test_hooks.log 2>&1
  judge $Q/generations/test_hooks $Q/generations/test_hooks/invalid > $L/judge_qwen_test_hooks.log 2>&1
  intervene --model $QW --directions train --split test --tag test_hooks_ext --extend-from test_hooks \
      --max-tokens 2048 --extend-tokens 6144 --max-model-len 16384 --skip-existing --candidates $CANDS \
      > $L/qwen_test_hooks_ext.log 2>&1
  judge $Q/generations/test_hooks_ext $Q/generations/test_hooks_ext/invalid > $L/judge_qwen_test_hooks_ext.log 2>&1
  $PY scripts/summarize.py $Q/generations/test_hooks --cont $Q/generations/test_hooks_ext \
      --csv $Q/analysis/test_hooks.csv > $L/summary_qwen_test_hooks.log 2>&1
}

step_capability() {
  failed capability && return
  for ds in math500 aime25; do
    S=1; [ $ds = aime25 ] && S=8   # 30 problems: 8 samples each
    $PY scripts/capability.py --gpu $G --model $OURO --dataset $ds --samples $S --tag cap --skip-existing \
        --candidates none ablate:v4_baseline:t3.l16 ablate:v4_baseline:t3.l16:apply=2 ablate:v4_baseline:t3.l16:apply=0 \
        actadd:v4_baseline:t3.l16:c=-1 > $L/ouro_capability_$ds.log 2>&1
    $PY scripts/capability.py --gpu $G --model $NB --dataset $ds --samples $S --tag cap --skip-existing \
        --candidates none ablate:v4_baseline:t1.l15 ablate:v4_baseline:t1.l15:apply=0 ablate:v4_baseline:t1.l15:apply=1 \
        actadd:v4_baseline:t1.l15:c=$NB_C > $L/nanbeige_capability_$ds.log 2>&1
  done
  $PY -I - > $L/summary_capability.log 2>&1 <<'EOF'
import glob, pandas as pd, os
rows = []
for f in sorted(glob.glob("data/runs/*/capability/*/cap/*.parquet")):
    df = pd.read_parquet(f); p = f.split("/")
    rows.append({"model": p[2], "dataset": p[4], "candidate": df.candidate.iloc[0], "accuracy": df.correct.mean(),
                 "closed": df.closed.mean(), "mean_tokens": df.n_tokens.mean(), "n": len(df)})
t = pd.DataFrame(rows); print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
t.to_csv("data/runs/capability_summary.csv", index=False)
EOF
}

for s in "$@"; do "step_$s"; echo "[$(date -u +%FT%TZ)] step $s finished" >> $L/followups_progress.log; done
