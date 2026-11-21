# Inkling 2 Mini: status and how to resume

Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, rev `6e7c824`): looped MoE, prelude layers 0-5, core 6-17 run 3
times (passes), coda 18-23; block attention residuals instead of a single residual stream; every pass restarts from the
prelude's state and reads the previous pass through gated cross-pass messages. Variable-pass research checkpoint:
`thinkingmachines/TML-d24-varloop-research` (rev `767c9d0`, 1-3 passes). Agreed choices (Nov 20 2026):

- NVFP4 / MXFP8 weights as shipped, through vLLM only (bf16 only if NVFP4 fails); separate environment
  `.venv-inkling` (`envs/inkling/README.md`).
- The model as released: official chat template (reasoning effort 0.9), temperature 1.0, top_p 1.0. This differs from
  the 0.6 used for the other models.
- Fixed-loop model first: steer single passes, passes 1+2, all passes (and everything incl. prelude / coda).
- Priority: refusal steering and kicks, then persona, then weight edits (on hold).

## Code

- `src/loop_steer/vllm_inkling.py`: interventions inside vLLM's `Inkling2LoopModel`, CUDA-graph safe (read its
  docstring: sites, units, spec kinds). resid_pre at a site = the running partial block as the layer visit starts.
  Checks run (scratch tests, Nov 20): ablation removes the direction exactly where selected and nowhere else; actadd adds
  exactly c * vec at its site; graph vs eager decodes identical under every intervention; nothing installed = plain
  vLLM exactly. Known: an ablated input is normalized unfused (~0.13 nat/token of rounding noise through the NVFP4
  experts); the random-direction control runs through the same path.
- `scripts/run_select.sh inkling TAG CANDIDATES...` (generation in `.venv-inkling`, judging in `.venv`);
  `scripts/inkling_directions.sh` (train generations -> directions), `scripts/inkling_refusal.sh` (selection, site
  pick, ablation grid), `scripts/inkling_actadd.sh` (actadd tuning + grid), `scripts/inkling_bundle.sh` (GCS move).
- `extract_activations.py`, `stream_norms.py`, `intervene.py`, `capability.py` handle Inkling.

## Done (data in `data/runs/Inkling-2-Mini`)

- Train generations (500 prompts x 3 CoTs x 3 answers): refusal 0.75; labels 1059 refusing / 279 complying CoTs.
- Directions `directions/train.pt` at all 48 sites (v4_cot, v4_baseline, hh_eoi, ...); `stream_norms.pt`.
- Selection (50 prompts, ablation in every unit; `analysis/sel.csv`): clean 0.73, random 0.75; **picked
  `v4_cot:t2.l11`** (0.12; CoTs ~2x longer, 5% capped); runner-up `hh_eoi:t0.l14` (0.18). Several directions at
  pass-1/2 layers 12-14 and coda layer 19 *raise* refusal to 0.89-0.99 when ablated (coherent refusals of borderline
  prompts). `analysis/inkling_site.json`.
- Kicks on MATH-500 (100 problems, per-problem random direction at layer 12 of each pass; `capability/math500/kickp`):
  clean 0.90; eps 1.5: pass 1 0.80, pass 2 0.49, pass 3 0.71 (damage = unclosed / very long reasoning).
- Ablation grid on the test set (250 prompts; `generations/grid_test`): all 8 candidates generated (none, random,
  passes 1 / 2 / 3, 1+2, all passes, everything), **not yet judged**.

## To do, in order

1. Judge + summarize the ablation grid:
   `.venv/bin/python scripts/judge.py --gpu 0 $R/generations/grid_test $R/generations/grid_test/invalid` then
   `.venv/bin/python scripts/summarize.py $R/generations/grid_test --max-tokens 4096 --csv $R/analysis/grid_test.csv`
   (`R=data/runs/Inkling-2-Mini`).
2. `GPU=1 scripts/inkling_actadd.sh`: actadd tuning (selection set + MATH-500), coefficient pick, actadd grid.
3. Kick KL (teacher-forced, `kick_kl.py` needs a vLLM path for Inkling), E13-style independent per-pass kicks, kicks in
   the prelude / coda; rest of the test set (offset 250, as E12); +u on harmless prompts (E14/E15).
4. Persona vectors (E5/E6/E11).
5. Inkling-specific: rebuild after ablation via messages vs prelude re-injection (ablate in passes 1-2, then zero /
   mean the messages or the prelude state), message-only steering, variable-pass checkpoint at 1/2/3 passes.
6. Weight edits (held).

## New instance

```bash
git clone git@github.com:aamixsh/loop_steer.git && cd loop_steer        # main
scripts/setup_remote.sh --cache-dir /workspace/cache                    # main .venv, judge model, prompts
source /workspace/cloud-sync/env.sh && scripts/inkling_bundle.sh pull    # wheel, Inkling runs, checkpoints, .venv-inkling
```

The vLLM wheel in the bundle is built for sm_120 (RTX PRO 6000 Blackwell). On other GPUs rebuild it
(`envs/inkling/README.md`, ~1 h) or wait for a vLLM release with vllm-project/vllm#69028.
