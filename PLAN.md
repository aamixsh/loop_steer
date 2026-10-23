# Plan

Edit this file in place (no running log): it holds the **current** plan and the **Next TODO** list, so work can be
picked up after a context reset. Results live in the README, `docs/presentation.md` and `results/`; machine setup in
`docs/handoff.md`.

## Goal

Refusal-direction ablation and steering for looped reasoning models (Ouro-1.4B-Thinking, 4 loops; Nanbeige4.2-3B,
2 loops), reproduced first on Qwen3-8B, and checked on an MoE model (gpt-oss-20b). Current focus: **where in the loops and layers the refusal decision is made**,
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
- **gpt-oss-20b (MoE), test set:** clean 92%, random 92%, hook ablation (`v4_baseline` L15) 2.1%, weight ortho over all
  32 experts 1.8% (biases kept, as upstream: 1.6%), actadd L13 c=-2 14.6%; 0.2-1% unclosed. Biases matter for an exact
  ablation (6% of `u` stays in the stream without them), not for behaviour. CoT-window directions and any direction from
  L7-11 break generation (screen). Judge = gpt-oss itself; refusal-phrase check agrees. Capability (8192 cap):
  MATH-500 88.8 clean / 86.0 hooks / 85.0 ortho / 65.4 actadd; AIME 52.5 / 40.8 / 38.8 / 10.8. Ablation lengthens
  CoTs (+25% tokens) so more hit the cap; accuracy of closed CoTs unchanged. AIME at 16k (tag `cap16k`): 65.4 / 57.1 /
  57.5 (gap 8 points, 25% vs 34% of CoTs still capped). Over-refusal 0.4-3.2% (clean 1.8%).
- Ouro direct answers (no thinking): 20% clean, 11-15% under ablation on the test set (the selection-set rise to
  15-18% was not confirmed).

## Running now

**Dory (NVIDIA Nemotron-Labs Dory 1.0 4B, looped nGPT)**: clean baselines queued 2026-10-22 19:32 ET (test set `none`
in TAG `test`; MATH-500 and AIME 2025 `none` in TAG `cap`); `validate`, `clean`, `directions` and three screen rounds
(32 candidates, `analysis/select_screen.csv`) are done. Which interventions go to the test set is open (see below).
- **Dory resists single-direction ablation** (50 selection prompts, all_refusal / unclosed): clean 89%, random 91%.
  `v4_baseline` hooks, u from: t0.l26 43% / 16%, t0.l22 47% / 13% (per-loop directions 38% / 12%), t0.l38 59% / 10%,
  t1.l38 62% / 10%, t1.l30 77%, t0.l18-20 78-82%, t0.l30 81%, later sites 74-94%, input l10-14 89-92%, and **t0.l24 /
  t0.l34 99%** (above clean). `v4_cot` 84-92% (no effect, CoTs intact, unlike Ouro and gpt-oss). Actadd t0.l22 c=-1 /
  -1.5 / -2: 73 / 62 / 52% (4 / 8 / 23% unclosed); t1.l30 c=-2 26% with 61% unclosed (broken). Weight edit: t0.l22 70%,
  t1.l38 77% plain, 78% lr-aware (hooks at the same sites 47% / 62%: the predicted leak).
- **Why: zero is not "complying" along u, and removing u is not enough.** Which u works does not track AUC (rank corr.
  -0.11 over 16 sites) but where 0 lies along it (CoT-window class midpoint over sites: -0.95). Affine ablation (ACE,
  Marshall et al. 2024; `ace=<method>` in intervene.py, hooks in both engines) sets the component to the complying
  class mean instead of 0, so the classes no longer differ along u anywhere: with the CoT-window reference it gives
  64-89% (u from t0.l22 78%, t0.l24 75%, t0.l26 64%, t0.l34 89%, t0.l38 73%, t1.l38 77%; <= 6% unclosed; random 92%).
  t0.l24 / t0.l34 drop from 99% (plain zeroing pushed toward refusal there). Lower refusal needs pushing past complying:
  plain zeroing at t0.l22 / t0.l26 (shift 1.5x the class gap) 47% / 43%; the end-of-prompt reference at t0.l22 (shift
  2.5x the gap) 26% with 19% unclosed. Over all 24 `v4_baseline` hook candidates, refusal follows the shift of the
  refusing CoT mean along u (rank corr. 0.74). So in Dory the drop comes from steering along u, not from removing it;
  refusal is largely carried outside u.
- The direction rotates more across depth than in Ouro: u from t1.l38 has cos > 0.5 with 48/112 sites (Ouro's u from
  t3.l16: 70/96), down to 0.16 in the second half (Ouro 0.55).
- Clean train run: 92.4% refusal (4,446 judged answers), 1.2% unclosed; labels 1,362 refusing / 94 complying CoTs
  (prompt level 457 / 33, 31 prompts mixed).
- Directions at 112 sites (16 input + 40 recurrent x 2 loops + 16 output sublayers; input/output report loop 0).
  End-of-prompt AUC rises through recurrent loop 0 and peaks at 0.95-0.96 at recurrent 34-42 in both loops (random
  0.5-0.65), flat through the output stack; the CoT window is 0.95 everywhere but random directions reach 0.8-0.96 there
  (as in Ouro and gpt-oss).
- Screen candidates are listed in `scripts/run_dory.sh` (step `screen`); full table in
  `analysis/select_screen.csv`.
- `validate`: greedy HF~vLLM prefix agreement 0.83 clean, 0.60 under random-direction ablation (above HF~HFclean 0.50
  and vLLM~vLLMclean 0.28, so both engines make the same change; bf16 greedy noise); actadd outputs match for the first
  ~20 tokens.
- No transformers or vLLM release supports Dory: `loop_steer.dory` (config + HF implementation, follows NVIDIA's vLLM
  PR #60149) and `loop_steer.vllm_dory` (that PR ported to vLLM 0.18, in-process) are ours. CPU checks (fp32, tiny
  random config, the PR's attention harness): port vs HF 4e-7; ablation hooks vLLM vs HF logits 6e-8 (ablation moves
  them 0.06), u.h < 4e-8 at every block input in both; weight edit vLLM vs HF 6e-8, restore exact; affine ablation
  vLLM vs HF 7e-8, every block input at its reference to 1e-7.
- Prediction: weight ortho leaks at every sublayer (the stream update uses a per-dimension interpolation weight), hooks
  stay exact. The tiny random model shows it: ortho leaves |u.h| 0.01-0.05 at block inputs (clean 0.15-0.17), lr-aware
  ortho 0.01-0.03.
gpt-oss runs use a bf16 copy (`data/models/gpt-oss-20b-bf16`). GPU 1 belongs to another project. gpuq kills a job silent
for `--idle` seconds (default 1800).

## Next TODO (in order)

1. [x] **Why late loops?** `scripts/loop_rebuild.py` (teacher-forced clean traces, separation of refusing vs complying
   along depth). Later loops rebuild the signal ablated upstream: prompt-end separation in Ouro's last loop is 93% of
   clean after ablating loop 1 only, 75% after loop 2, 37% after loop 3, 19% after loop 4, 10% after all. Nanbeige:
   59% after loop 1, 15% after loop 2. This ordering matches test-set refusal. Add to the deck (figure:
   `data/runs/<model>/figures/loop_rebuild_*.png`).
2. [x] **1- and 2-loop Ouro weight edits** (selection set, tags `loops2`, `loops1`). 1-loop Ouro does not work even
   clean (96% unclosed). At 2 loops: plain ortho with a random direction still breaks generation (leak 1); norm-aware
   ortho with the refusal direction gives 10% refusal like hooks (clean 40%) but 54% unclosed vs 38% for hooks,
   whereas at 4 loops it leaves 99% unclosed (leak 2 compounds with loops). Add to the deck's "Testing the
   explanation"; continuing the capped CoTs (2-loop Ouro thinks longer) would sharpen the comparison.
3. [ ] **Read the position-restricted runs** (done, selection set, tag `select_pos` for Ouro and Nanbeige): which token
   classes carry the effect? Confirm the main ones on the test set (`run_test_set.sh` takes the same candidates).
   `pos=` is vLLM-only; validated by `scripts/validate_position_gates.py` (exact).
3b. [x] **gpt-oss (MoE)**: validation, screen, test set, capability, over-refusal done (deck: MoE slides). Left: the
   position-restricted ablation on gpt-oss (Harmony bounds) once item 3 is read; AIME at 32k (does the gap close?);
   an independent judge on a sample of gpt-oss answers (gpt-oss currently judges itself).
4. [ ] Bootstrap confidence intervals over prompts in `summarize.py` / `loop_sweep_summary.py`.
5. [ ] Capability at a longer cap (AIME 2025 at 16-32k tokens) and actadd at matched strength across models.
6. [ ] Explain the Qwen whole-CoT gap (15% vs the paper's 7.5%): train set size, 3 vs 5 answers, top_p.
7. [ ] `paired_cot` and actadd-coefficient sweeps on the looped models; a teacher-forced per-loop screen for Ouro.
8. [ ] Retire `queue_*.sh` and `run_followups.sh` (keep `run_followups.sh` as the record of how the follow-ups ran).
9. [ ] Render the Marp deck (no `marp` on this machine) and check slide overflow.
10. [ ] Housekeeping before making the repo public: license, decide on rewriting git history.
11. [ ] **Dory** (see Running now): validate, clean run, directions, three screen rounds (done), clean baselines
    (queued), then test set, capability; compare the leak prediction with `ortho_compare`-style measurements (needs a
    Dory branch: no `model.norm`, per-sublayer `lr`).

## Open questions for the user

- Selection-set-only results remain (all-subsets sweep, loop x layer map, Ouro weight edit). Rerun any of them on the
  test set before reporting?
- GPU use: GPU 0 only so far (GPU 1 runs another project's jobs).
- Dory test set: by the usual rule (lowest refusal with unclosed <= ~10%) the hook winner is u from t0.l38 (59%);
  the strongest are t0.l26 (43%, 16% unclosed) and affine t0.l22 with the end-of-prompt reference (26%, 19%). Which to
  run (each ~1 h on the test set), and whether to test multi-direction (subspace) ablation first?

## How to resume

```bash
source scripts/env.sh && scripts/verify_setup.sh --no-gpu           # environment and data check
nvidia-smi                                                            # what runs
scripts/gpuq.py list                                                  # GPU 0 job queue
scripts/gpuq.py add -n NAME -- .venv/bin/python scripts/X.py --gpu '$GPU' ...   # queue a job behind the running ones
GPU=0 scripts/run_test_set.sh ouro|nanbeige|qwen|gptoss TAG CANDIDATE ...  # test-set run with continuations + summary
GPU=0 ACTADD_C=-1 VARIANT=shared|perloop scripts/run_loop_sweep.sh   # loop-subset sweep (MODEL_KEY=ouro|nanbeige)
scripts/loop_sweep_summary.py loop_sweep loop_sweep_perloop --model Ouro-1.4B-Thinking
```

Conventions: candidates are `kind:method:site[:c=..][:apply=..][:dir=perloop][:na=1]` (docstring of
`scripts/intervene.py`); loops are 0-based in code and 1-based in tables and slides; everything is skip-existing.
