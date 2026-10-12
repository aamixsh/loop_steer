#!/usr/bin/env bash
# Test-set run (487 prompts, 3 CoTs x 3 answers) of intervention candidates for one model: generate, judge, continue the
# CoTs that hit the cap to 8192 tokens in total, judge those, and summarize with the continuations merged.
#   GPU=0 scripts/run_test_set.sh ouro|nanbeige|qwen|gptoss TAG CANDIDATE [CANDIDATE ...]
# Candidates use the syntax of scripts/intervene.py. Results: data/runs/<model>/generations/TAG (+ TAG_ext) and
# analysis/TAG.csv. Skip-existing, so it can be re-run to resume or to add candidates to an existing TAG (the summary then
# covers all of the TAG's candidates). DRY=1 prints the commands.
set -u -o pipefail
cd "$(dirname "$0")/.."
KEY=${1:?model key: ouro|nanbeige|qwen|gptoss}; TAG=${2:?tag}; shift 2
[ $# -gt 0 ] || { echo "no candidates" >&2; exit 2; }
PY=.venv/bin/python; G=${GPU:-0}; L=data/runs/logs
case $KEY in
  ouro)     MODEL=ByteDance/Ouro-1.4B-Thinking; RUN=Ouro-1.4B-Thinking; CAP=2048; GEN=""; EXT="--extend-tokens 6144 --max-model-len 16384" ;;
  nanbeige) MODEL=Nanbeige/Nanbeige4.2-3B; RUN=Nanbeige4.2-3B; CAP=4096; GEN="--max-tokens 4096 --max-model-len 12288"
            EXT="--extend-tokens 4096 --max-model-len 16384" ;;
  qwen)     MODEL=Qwen/Qwen3-8B; RUN=Qwen3-8B; CAP=2048; GEN=""; EXT="--extend-tokens 6144 --max-model-len 16384" ;;
  gptoss)   MODEL=openai/gpt-oss-20b; RUN=gpt-oss-20b; CAP=4096; GEN="--max-tokens 4096 --max-model-len 12288"
            EXT="--extend-tokens 4096 --max-model-len 16384" ;;
  *) echo "unknown model key $KEY" >&2; exit 2 ;;
esac
D=data/runs/$RUN/generations
exec 3>&1   # progress lines go to the job's stdout; command output is tee'd to its own log AND the job's stdout, so a
            # queue watchdog watching the job's output sees activity
run() { echo "+ $*" >&3; [ -n "${DRY:-}" ] || "$@"; }
log() { if [ -n "${DRY:-}" ]; then echo /dev/null; else echo "$L/$1"; fi; }
judge_dirs() { local a=(); for d in "$@"; do [ -d "$d" ] && a+=("$d"); done; run $PY scripts/judge.py --gpu $G "${a[@]}"; }

run $PY scripts/intervene.py --gpu $G --model $MODEL --directions train --split test $GEN --tag $TAG --skip-existing --candidates "$@" 2>&1 | tee "$(log ${KEY}_$TAG.log)" >&3 || echo "generation step failed (see $L/${KEY}_$TAG.log)"
judge_dirs $D/$TAG $D/$TAG/invalid 2>&1 | tee "$(log judge_${KEY}_$TAG.log)" >&3
run $PY scripts/intervene.py --gpu $G --model $MODEL --directions train --split test --tag ${TAG}_ext --extend-from $TAG \
    --max-tokens $CAP $EXT --skip-existing --candidates "$@" 2>&1 | tee "$(log ${KEY}_${TAG}_ext.log)" >&3 || echo "continuation step failed"
judge_dirs $D/${TAG}_ext $D/${TAG}_ext/invalid 2>&1 | tee "$(log judge_${KEY}_${TAG}_ext.log)" >&3
[ -n "${DRY:-}" ] || mkdir -p data/runs/$RUN/analysis
run $PY scripts/summarize.py $D/$TAG --max-tokens $CAP --cont $D/${TAG}_ext --csv data/runs/$RUN/analysis/$TAG.csv 2>&1 | tee "$(log summary_${KEY}_$TAG.log)" >&3
