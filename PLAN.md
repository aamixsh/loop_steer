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

**Self-correction experiments** (queued 2026-11-06 by `scripts/queue_selfcorr.sh`, both GPUs; all HF unless noted;
outputs under `data/runs/<model>/analysis/` and the model's run dir):
- `perturb_response.py` (GPU 1): impulse response (eps x |h| x v added at one site; v = u, 8 random, or `--vec`) traced to
  the output, loop contraction by power iteration, clean drift between loops. Ouro, Nanbeige, Dory, Qwen and gpt-oss
  (dense: 4 blocks as virtual loops); extra loops (Ouro 8, Nanbeige 4, Dory 6); CAA and persona vectors as v.
  Smoke test (Ouro, 8 prompts, eps 0.25): random kick at loop 1 L0 -> 0.002 KL at the output, at loop 4 L0 -> 0.16.
- `loop_patch.py` (GPU 1): swap the stream entering loop k with a length-matched harmful/harmless prompt's (last token,
  prompt-end tokens, all positions); recovery of u and of the whole state. Smoke (Ouro, 8 pairs): last-token swaps
  recover 0.9-1.0; prompt-end swaps into loop 2 recover u 0.95 but the state only 0.5; all-position swaps 0.
- `loop_rebuild.py --blocks 4` (GPU 1): dense control for the rebuild (Qwen, gpt-oss); Dory rebuild (train traces).
- `caa_loops.py` (GPU 1): 7 behaviour A/B sets (Rimsky et al.), vectors at every site (`directions/caa.pt`), steering
  in the first / last / every loop, P(matching answer).
- `neuron_push.py` (GPU 0, vLLM): Neuron-Persona-Library-style pushes of 40 random neurons per MLP layer (x2 signs x
  strengths 0.5, 1.0 of the median stream norm) in the first / last / every loop; gpt-oss labels non-AI identity claims.
  Uses the new per-request `actadd_table` hook spec.
- `persona_vectors.py` (GPU 0, vLLM + HF): 7 traits (Chen et al. 2025 data), vectors over answer tokens
  (`directions/persona.pt`), steering at 3 layers x first / last / every loop x c = 0.05-0.2 of the stream norm, judged
  for trait and coherence. Ouro breaks at c >= 0.25.
- Ouro has no thinking switch: `selfcorr.no_think_prompt` gives it a one-line closed thought (empty block -> it ends
  its turn at once); identity prompts use its plain template (it answers them directly).

**Dory (NVIDIA Nemotron-Labs Dory 1.0 4B, looped nGPT)**: nothing running. **Test set** (487 prompts, `hh_eoi` u from
t0.l26, capped CoTs extended to 8192): clean 92.3%, random 91.4%, **hooks 12.4%** (14% of CoTs hit 4096, 0.9% still
unclosed at 8192), weight edit 25.5% (1.4%), actadd c=-1 23.1% (3.4%), hooks with decay=200 41.1% (0.5%). Capability
(CoT <= 8192): MATH-500 clean 77.2%, hooks 75.0%, weight edit 74.6%, actadd c=-1 82.0% (shorter CoTs); AIME 2025
28.3 / 25.4 / 26.3 / 26.7%. Over-refusal 0.8-2.8% (clean 2.4%).
- **Loop subsets at 4 loops** (selection set, TAG `loops4`, `scripts/run_select.sh` with N_LOOPS=4; u = hh_eoi t0.l26;
  `apply=` now counts recurrent loops only, Dory's input / output stacks are loop -1): every set containing loop 4
  gives 10-15% all_refusal (17-20% capped), every set without it 86-93% (loop 1 92%, loops 1-3 86%, in/out only 93%);
  clean 90%, random 93%; actadd c=-1 all loops 20%, loop 1 only 92%. Same at 2 loops (TAG `loop_gate`): loop 2 alone 16% (16% capped), loops 1+2 10%, everything 8% (screen); loop 1
  alone 84%, in/out only 91%, actadd c=-1 loop 1 only 84%; clean 90%.
  Next: input stack + loop 1 (`apply=-1,0`), leave-one-out (`0,1,3`, `0,2,3`), ablation at the site only / final
  stream only (needs a `layers=` option), u's separation in loops 3-4 at 4 loops, and the actadd(+u)-on-harmless
  check of the direction; `loop_rebuild.py` Dory branch.
- **The harmful - harmless direction works** (`scripts/run_dory.sh contrast`: 1,409 harmful train prompts minus the
  selection set vs 1,174 harmless, end-of-prompt tokens; split-half 1.00 everywhere). Screen: hooks with u from t0.l26
  8.4% refusal / 19% capped (t0.l22, t0.l38, t1.l38: 6-10% / 31-33%), affine 1-2% / 31-39%, l62 75%; actadd c=-1 21%.
  The capped CoTs are coherent (the full answer written inside the thinking), except the affine variant (some
  degenerate into random strings). The 33-prompt end-of-prompt direction below was too noisy.
- **Decaying steering** (`decay=N`: full on the prompt, half-cosine to 0 over the first N CoT tokens; vLLM hooks):
  decay=200 cuts capped CoTs (hh hooks 19% -> 14%, `v4_baseline` hooks 16% -> 9%, actadd 23% -> 9%) but loses most of
  the effect (hh hooks 8% -> 38% refusal): the CoT tokens carry most of it.
- `v12_cot150` (first 150 CoT tokens, AUC 0.96-0.97, split-half 0.95) does not work on Dory: hooks 82-100% refusal.
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

**Paper plan:** `docs/paper_plan.md` lists the experiments that must run before publishing (one setup for all looped
models, seeds and intervals, dense controls for self-correction, causal tests of the norm mechanism, attacks). Pick
work from there; the items below are the older queue.

1. [x] **Why late loops?** `scripts/loop_rebuild.py` (teacher-forced clean traces, separation of refusing vs complying
   along depth). Later loops rebuild the signal ablated upstream: prompt-end separation along u at Ouro's last layer
   is 94% of clean after ablating loop 1 only, 78% after loop 2, 49% after loop 3 (0% when the last loop is ablated).
   Nanbeige: 67% after loop 1. This ordering matches test-set refusal. In the deck (`figures/rebuild.png`):
   after a loop-1 ablation Ouro's loop 2 rebuilds 4% -> 83% of the separation along u (Nanbeige 0% -> 67%); the
   inter-loop norm contributes 4% / 0%, so the rebuild is recomputation, not the norm.
2. [x] **1- and 2-loop Ouro weight edits** (selection set, tags `loops2`, `loops1`). 1-loop Ouro does not work even
   clean (96% unclosed). At 2 loops: plain ortho with a random direction still breaks generation (leak 1); norm-aware
   ortho with the refusal direction gives 10% refusal like hooks (clean 40%) but 54% unclosed vs 38% for hooks,
   whereas at 4 loops it leaves 99% unclosed (leak 2 compounds with loops). Add to the deck's "Testing the
   explanation" (done); continuing the capped CoTs (2-loop Ouro thinks longer) would sharpen the comparison.
3. [x] **Read the position-restricted runs** (selection set, tag `select_pos`, in the deck): prompt-only ablation does
   nothing (Ouro 64%, Nanbeige 74%), CoT only 22% / 44%, prompt + CoT 9% / 19% (all tokens 8% / 16%). Test-set
   confirmation is in `docs/paper_plan.md` (B.3).
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
