#!/usr/bin/env bash
# Selection-set run (50 held-out train prompts, offset 500; 3 CoTs x 3 answers) of intervention candidates for one
# model: generate, judge, summarize to analysis/TAG.csv. No CoT continuation (unclosed CoTs are reported as such).
#   GPU=0 [N_LOOPS=2] scripts/run_select.sh ouro|nanbeige|qwen|gptoss TAG CANDIDATE [CANDIDATE ...]
# N_LOOPS runs a looped model with that many loops (intervene.py --n-loops; use its own TAG). Skip-existing, so it can be
# re-run to resume or to add candidates to a TAG. Output of every step goes to its log in data/runs/logs AND to stdout
# (so a queue watchdog sees progress). DRY=1 prints the commands.
set -u -o pipefail
cd "$(dirname "$0")/.."
KEY=${1:?model key: ouro|nanbeige|qwen|gptoss}; TAG=${2:?tag}; shift 2
[ $# -gt 0 ] || { echo "no candidates" >&2; exit 2; }
PY=.venv/bin/python; G=${GPU:-0}; L=data/runs/logs
case $KEY in
  ouro)     MODEL=ByteDance/Ouro-1.4B-Thinking; RUN=Ouro-1.4B-Thinking; CAP=2048; GEN="" ;;
  nanbeige) MODEL=Nanbeige/Nanbeige4.2-3B; RUN=Nanbeige4.2-3B; CAP=4096; GEN="--max-tokens 4096 --max-model-len 12288" ;;
  qwen)     MODEL=Qwen/Qwen3-8B; RUN=Qwen3-8B; CAP=2048; GEN="" ;;
  gptoss)   MODEL=openai/gpt-oss-20b; RUN=gpt-oss-20b; CAP=4096; GEN="--max-tokens 4096 --max-model-len 12288" ;;
  *) echo "unknown model key $KEY" >&2; exit 2 ;;
esac
[ -n "${N_LOOPS:-}" ] && GEN="$GEN --n-loops $N_LOOPS"
D=data/runs/$RUN/generations
exec 3>&1
run() { echo "+ $*" >&3; [ -n "${DRY:-}" ] || "$@"; }
log() { if [ -n "${DRY:-}" ]; then echo /dev/null; else echo "$L/$1"; fi; }

run $PY scripts/intervene.py --gpu $G --model $MODEL --directions train --split train --offset 500 --n-prompts 50 $GEN \
    --tag $TAG --skip-existing --candidates "$@" 2>&1 | tee "$(log ${KEY}_$TAG.log)" >&3 \
    || echo "generation step failed (see $L/${KEY}_$TAG.log)"
dirs=(); for d in $D/$TAG $D/$TAG/invalid; do [ -d "$d" ] && dirs+=("$d"); done
run $PY scripts/judge.py --gpu $G "${dirs[@]}" 2>&1 | tee "$(log judge_${KEY}_$TAG.log)" >&3
[ -n "${DRY:-}" ] || mkdir -p data/runs/$RUN/analysis
run $PY scripts/summarize.py $D/$TAG --max-tokens $CAP --csv data/runs/$RUN/analysis/$TAG.csv 2>&1 \
    | tee "$(log summary_${KEY}_$TAG.log)" >&3
