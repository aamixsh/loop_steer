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
  models; test-set results (487 prompts) exist for the best candidates and the loop configs (capped CoTs continued to
  8192 tokens). The deck (`docs/presentation.md`) and README hold every result; `results/` holds the tables.
- Qwen works with weight ortho (6%), hook ablation (5.5%) and actadd (6.8%) against 61% clean.
- Ouro: hooks 10%, actadd 17% (clean 61%); weight ortho breaks generation (sandwich norms + amplifying inter-loop
  norm; norm-aware ortho still 99% unclosed). Nanbeige: hooks 16-18%, weight ortho 11%, actadd 26% (clean 78%).
- **Loops (test set):** Ouro ablation in loop 1/2/3/4 only: 49/28/18/20%, loops 3+4 13%, all 10%; Nanbeige loop 1/2
  only 53/28%, both 18%. Later loops carry most of the effect, no single loop is enough; the selection set was
  optimistic by 5-10 points. Ablation needs a late loop; actadd is a dose over loops (selection set, all subsets).
- **Layers** (Ouro actadd, selection): effect grows with loop index; in loop 4 only layers 4-12 work.
- **Collateral:** ablation costs no capability (MATH-500, AIME 2025 within noise); actadd costs MATH-500 3.6 (Ouro)
  and 8.8 points (Nanbeige), Nanbeige AIME 29.6% -> 13.8%. No over-refusal (0-3% on 250 harmless prompts).
- Ouro direct answers (no thinking): 20% clean, 11-15% under ablation on the test set (the selection-set rise to
  15-18% was not confirmed).

## Running now

GPU 0 queue (`scripts/gpuq.py list`), selection set, `scripts/run_select.sh`: Ouro with 2 and with 1 loop (tags
`loops2`, `loops1`: clean, hook ablation, plain and norm-aware weight ortho, refusal and random direction), then
position-restricted ablation on Ouro and Nanbeige (tag `select_pos`: `v4_baseline` on prompt / cot / answer /
prompt+cot / cot+answer tokens, `v4_cot` on each class, plus clean and full ablation). Results go to
`data/runs/<model>/analysis/<tag>.csv`. GPU 1 belongs to another project. The gpuq watchdog kills a job whose log is
silent for `--idle` seconds (default 1800): jobs must print progress to stdout.

## Next TODO (in order)

1. [x] **Why late loops?** `scripts/loop_rebuild.py` (teacher-forced clean traces, separation of refusing vs complying
   along depth). Later loops rebuild the signal ablated upstream: prompt-end separation in Ouro's last loop is 93% of
   clean after ablating loop 1 only, 75% after loop 2, 37% after loop 3, 19% after loop 4, 10% after all. Nanbeige:
   59% after loop 1, 15% after loop 2. This ordering matches test-set refusal. Add to the deck (figure:
   `data/runs/<model>/figures/loop_rebuild_*.png`).
2. [ ] **Read the 1- and 2-loop Ouro weight-edit runs** (queued): does norm-aware ortho with the refusal direction give
   valid CoTs and low refusal at 1-2 loops (as the mechanism predicts) while it breaks at 4? Then deck slide
   "Testing the explanation".
3. [ ] **Read the position-restricted runs** (queued, selection set): which token classes carry the effect? Confirm
   the main ones on the test set (`run_test_set.sh` takes the same candidates). `pos=` is vLLM-only; validated by
   `scripts/validate_position_gates.py` (exact).
4. [ ] Bootstrap confidence intervals over prompts in `summarize.py` / `loop_sweep_summary.py`.
5. [ ] Capability at a longer cap (AIME 2025 at 16-32k tokens) and actadd at matched strength across models.
6. [ ] Explain the Qwen whole-CoT gap (15% vs the paper's 7.5%): train set size, 3 vs 5 answers, top_p.
7. [ ] `paired_cot` and actadd-coefficient sweeps on the looped models; a teacher-forced per-loop screen for Ouro.
8. [ ] Retire `queue_*.sh` and `run_followups.sh` (keep `run_followups.sh` as the record of how the follow-ups ran).
9. [ ] Render the Marp deck (no `marp` on this machine) and check slide overflow.
10. [ ] Housekeeping before making the repo public: license, decide on rewriting git history.

## Open questions for the user

- Selection-set-only results remain (all-subsets sweep, loop x layer map, Ouro weight edit). Rerun any of them on the
  test set before reporting?
- GPU use: GPU 0 only so far (GPU 1 runs another project's jobs).

## How to resume

```bash
source scripts/env.sh && scripts/verify_setup.sh --no-gpu           # environment and data check
nvidia-smi                                                            # what runs
scripts/gpuq.py list                                                  # GPU 0 job queue
scripts/gpuq.py add -n NAME -- .venv/bin/python scripts/X.py --gpu '$GPU' ...   # queue a job behind the running ones
GPU=0 scripts/run_test_set.sh ouro|nanbeige|qwen TAG CANDIDATE ...       # test-set run with continuations + summary
GPU=0 ACTADD_C=-1 VARIANT=shared|perloop scripts/run_loop_sweep.sh   # loop-subset sweep (MODEL_KEY=ouro|nanbeige)
scripts/loop_sweep_summary.py loop_sweep loop_sweep_perloop --model Ouro-1.4B-Thinking
```

Conventions: candidates are `kind:method:site[:c=..][:apply=..][:dir=perloop][:na=1]` (docstring of
`scripts/intervene.py`); loops are 0-based in code and 1-based in tables and slides; everything is skip-existing.
