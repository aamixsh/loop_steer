# Plan

Edit this file in place (no running log): it holds the **current** plan and the **Next TODO** list, so work can be
picked up after a context reset. Results live in the README, `docs/presentation.md` and `results/`; machine setup in
`docs/handoff.md`.

## Goal

Refusal-direction ablation and steering for looped reasoning models (Ouro-1.4B-Thinking, 4 loops; Nanbeige4.2-3B,
2 loops), reproduced first on Qwen3-8B. Current focus: **where in the loops and layers the refusal decision is made**,
and why weight orthogonalization fails on Ouro.

## Where we are

- Pipeline, directions, interventions (hook ablation, weight ortho, actadd) and the StrongREJECT judge run on all three
  models. Test-set results (487 prompts) exist for the best candidates; capped CoTs are continued to 8192 tokens.
- Qwen works with weight ortho (6%), hook ablation (5.5%) and actadd (6.8%) against 61% clean.
- **Loops** (selection prompts, 50): ablation in the later loops carries the effect, loop 1 alone is weak, both for
  Ouro and Nanbeige. Each loop's own direction is **not** better than the loop-4 (last-loop) direction. On the test set
  Ouro ablation in loop 3 only leaves 18.5% refusal against 10.4% for all loops (clean 61%): the later loops carry most
  of the effect, not all of it.
- **Layers** (Ouro actadd, one loop and one layer): effect grows with loop index; in loop 4 only layers 4 to 12 work.
- **Weight ortho on Ouro**: the norm-aware edit is harmless for a random direction but still breaks generation with the
  refusal direction (99% unclosed CoTs), so the inter-loop norm is the main cause.
- No over-refusal increase (0 to 3% refusal on 250 harmless prompts under every config). Capability (MATH-500, Ouro so
  far): 80.6% clean, 80.4 to 81.6% under ablation, 77.0% under actadd c=-1 (-3.6 points, noise is about 2).
- Code from this session is committed; run data is in `data/` (not in git); summary tables are in `results/`.

## Running now (GPU 0 only; GPU 1 belongs to another project's jobs)

Everything runs through one FIFO runner, `scripts/gpuq.py` (state in `data/queue/gpu0`, logs in `data/runs/logs/gpuq/`;
`scripts/gpuq.py list` shows it). Queued in order: Nanbeige AIME 2025 (5 candidates x 8 samples, resumes after the saved
clean candidate), the full Ouro AIME 2025 run, then `capability_summary.py` (writes `data/runs/capability_summary.csv`).
The earlier chain (`queue_followups.sh`) was stopped after its MATH-500 and layer-map steps had finished. Add new jobs
with `scripts/gpuq.py add -n NAME -- CMD ...`; the runner starts them when the GPU is free.

Ouro is slow (about 55 min per 500 MATH-500 samples: 4 loops, 790 KB of KV cache per token so only about 35 sequences
fit, and hooks force eager mode), so expect about 5 hours for its AIME run. Known issue: AIME 2025 at the 8192-token CoT
cap leaves about 70% of Nanbeige's clean samples unclosed, so its accuracy mostly measures what finishes under the cap;
MATH-500 is the informative capability result.

## Next TODO (in order)

1. [ ] **When the capability jobs finish** (`gpuq.py list`: capability-summary done): `scripts/export_results.sh`, add the capability
   rows (MATH-500 now, AIME) to the README findings and the deck's caveats slide, commit `results/`, push.
2. [ ] Retire `queue_*.sh` and `run_followups.sh` in favour of `gpuq.py` jobs (keep `run_followups.sh` steps as the
   record of how the follow-up experiments were run, or turn them into gpuq job lists).
3. [ ] **Nanbeige test-set confirmation** of the loop configs (ablate loop 2 only, loop 1 only; actadd `c=-2`) with
   continuations, as done for Ouro (`run_followups.sh` step `confirm` is the template).
4. [ ] **Two-loop Ouro with norm-aware ortho** (the handoff prediction: leak 2 needs several passes). Needs a loop-count
   override for the vLLM engine (HF has `load_model(n_loops_override=...)`).
5. [ ] **Position-restricted ablation** (prompt tokens / CoT tokens / answer tokens) in `hooks.py` and
   `vllm_hooks.py`, then `v4_baseline` and `v4_cot` with each. HF backend can split prefill vs decode directly; vLLM
   needs per-request prompt lengths.
6. [ ] **Why ablation raises direct answers on Ouro** (9% to 15-18% of samples answer without thinking): CPU analysis of
   the existing generations (`direct_frac` in the sweep tables).
7. [ ] Confidence intervals (bootstrap over prompts) in `loop_sweep_summary.py`; differences below about 4 points are noise.
8. [ ] Explain the Qwen whole-CoT gap (15% vs the paper's 7.5%): train set size, 3 vs 5 answers, top_p.
9. [ ] `paired_cot` and actadd sweeps on the looped models; a teacher-forced per-loop screen for Ouro.
10. [ ] Housekeeping before making the repo public: license, decide on rewriting git history.

## Open questions for the user

- Loop-restricted configs are chosen on the same 50 selection prompts the directions were picked on; the Ouro test-set
  run showed loop 3 alone to be optimistic there. Run the remaining headline claims on the test set before reporting?
- GPU use: the follow-up queue used GPU 0 only; GPU 1 has been idle. Allow GPU 1 for parallel jobs?

## How to resume

```bash
source scripts/env.sh && scripts/verify_setup.sh --no-gpu           # environment and data check
cat data/runs/logs/followups_progress.log; nvidia-smi                 # what finished, what runs
scripts/gpuq.py list                                                  # GPU 0 job queue
scripts/gpuq.py add -n NAME -- .venv/bin/python scripts/X.py --gpu '$GPU' ...   # queue a job behind the running ones
scripts/run_followups.sh <step> ...                                   # steps: see the header of that script
GPU=0 ACTADD_C=-1 VARIANT=shared|perloop scripts/run_loop_sweep.sh   # loop-subset sweep (MODEL_KEY=ouro|nanbeige)
scripts/loop_sweep_summary.py loop_sweep loop_sweep_perloop --model Ouro-1.4B-Thinking
```

Conventions: candidates are `kind:method:site[:c=..][:apply=..][:dir=perloop][:na=1]` (docstring of
`scripts/intervene.py`); loops are 0-based in code and 1-based in tables and slides; everything is skip-existing.
