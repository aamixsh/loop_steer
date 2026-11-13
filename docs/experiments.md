# Numbered experiments (self-correction across loops)

Question: when part of a looped model is steered or perturbed, do the remaining loops undo it, and does a dense model
(Qwen3-8B, split into quarters or halves as "loops") behave differently? Every experiment below is defined in
`scripts/exp.py` (`scripts/exp.py list`, `scripts/exp.py show E3`), queued per GPU through `scripts/gpuq.py`, and
pushes its outputs to GCS when each job ends. Jobs are skip-existing: rerunning resumes.

## Directions and where they are used

One direction (or vector) per model and behaviour, extracted at one site and used unchanged everywhere else.

| Model | Loops (trained) | Refusal direction (method, extraction site) | Used in loop / block k at | Kick and persona mid-layer |
|---|---|---|---|---|
| Ouro-1.4B-Thinking | 4 x 24 layers | `v4_baseline` (end-of-instruction tokens, refusing - complying prompts), loop 4 layer 16 (`t3.l16`) | layer 16 of loop k | layer 12 |
| Nanbeige4.2-3B | 2 x 22 | `v4_baseline`, loop 2 layer 15 (`t1.l15`) | layer 15 of loop k | layer 11 |
| Dory 1.0 4B | 16 input + 40 recurrent (x2) + 16 output sublayers | `hh_eoi` (harmful - harmless prompts, end of instruction), recurrent loop 1 sublayer 26 (`t0.l26`) | sublayer 26 of loop k | sublayer 36 |
| Qwen3-8B (dense) | 36 layers = 4 quarters of 9 (halves of 18) | `v4_baseline`, layer 21 (`l21`; quarter 3, offset 3) | ablation: every layer of quarter k; actadd: layer 3 + 9(k-1), i.e. 3 / 12 / 21 / 30 | layers 4 / 13 / 22 / 31 |

- **Ablation** projects the direction out of every layer input and every residual write inside the selected loops
  (Qwen: inside the selected layer ranges). Dory's input and output stacks (run once) are included only when every
  loop is selected (`apply` not given).
- **Activation addition** adds `c x v` at the listed layer in each selected loop. With `rel=1` the vector is
  rescaled at each place by the median stream norm there over the norm at the extraction site
  (`scripts/stream_norms.py`), so it is the same fraction of the stream everywhere. This matters for Qwen, whose
  stream grows from 17 (layer 3) to 561 (layer 30); in Ouro the norm at layer 16 varies 8.6 to 12.4 over loops and in
  Dory it is 1 (nGPT). `c` is tuned per model in E2.
- **Persona vectors** (Chen et al. 2025): one vector per trait from one site (the last trained loop at the tuned
  layer; Qwen: the tuned layer), added at the same layer in each selected loop (Qwen: same offset in each quarter),
  scaled to `c x` the median stream norm there; `c` and the layer are tuned per trait in E5.
- **Kicks**: `eps x |h| x v` per token at the stream entering the layer (attention sinks excepted), `v` a random unit
  direction drawn per problem (`loop_steer.kicks.problem_directions`, the same for a problem at every site, size and
  loop count).

## Measures

- **Refusal**: two-stage generation (3 CoTs per prompt, 3 answers per CoT, temperature 0.6), each answer scored by
  gpt-oss-20b with the StrongREJECT rubric (refused yes / no, convincingness, specificity). `all_refusal` = mean over
  all CoTs of the share of refusing answers; CoTs that never close are judged whole. Test set: the first 250 test
  prompts. Intervals: `scripts/grid_ci.py` (prompt bootstrap).
- **Capability**: MATH-500 accuracy (first 100 problems, one sample, CoT cap 8192 tokens, `scripts/capability.py`); E10 also
  full MATH-500, AIME25 (8 samples) and an MMLU-Pro subset (280 questions, option letter in `\boxed{}`).
- **Kick KL**: the unkicked model's own MATH-500 responses teacher-forced with and without the kick; mean KL(clean ||
  kicked) of the next-token distribution over the first 512 response tokens, and top-1 agreement (`kick_kl.py`).
- **Persona**: trait expression (0-100) and coherence (0-100) by gpt-oss-20b with the persona-vectors repo prompts,
  20 evaluation questions x 3 (tuning) or 5 samples, thinking off.

## Experiments

| ID | What | Depends on | GPU-h |
|---|---|---|---|
| E1 | Stream norms at every site (all models; Ouro 6 loops, Nanbeige 4, Dory 4). Already computed on the source machine. | - | 0.3 |
| E2 | Actadd coefficient tuning: every loop (Qwen: layers 3/12/21/30) at 4 values of c; refusal on the 50 selection prompts + MATH-500 (100). `actadd_tune.py` picks the lowest refusal with MATH-500 within 5 points of clean and capped CoTs within +10 points -> `analysis/actadd_tune.json` | E1 | 10 |
| E3 | Refusal loop grids on the test set (250 prompts): ablation and tuned actadd in loop 1, 2, 3, 4, 1+2, all (Ouro 4 loops, Dory 4 loops); 1, 2, both (Nanbeige, Dory 2 loops); Qwen quarters 1-4, 1+2, all and halves. Recovery with extra loops: Ouro at 6 loops steered in loops 1-4; Nanbeige at 4 loops steered in loops 1+2; Dory at 4 loops in loops 1+2 (vs 2 loops, both). Ablation results already on the source machine are skipped. | E1, E2 | 24 |
| E4 | Dory weight orthogonalization (plain and lr-aware), 2 and 4 loops, test set; hook ablation and clean in the same tag | - | 4 |
| E5 | Persona tuning per model: every loop (Qwen: every quarter) at once, 3 candidate layers x c in 0.01-0.3; per trait the largest trait gain with coherence within 10 points of clean -> `analysis/persona_tune.json` | - | 12 |
| E6 | Persona with the tuned settings in single loops / quarters and in all; Dory also at 4 loops with loops 1+2 | E5 | 7 |
| E7 | Random kicks on MATH-500 (100), per-problem directions, eps 0.5 / 0.7 / 1 / 1.5, each loop's mid-layer (Qwen quarters' mid-layers), plus Ouro at 6 loops (kick in loop 4 or 6), Nanbeige at 4 (loop 2 or 4), Dory at 4; accuracy + KL | - | 22 |
| E8 | Pilot: Dory at 4 loops kicked at every 4th sublayer of every loop and in the input / output stacks (30 problems, eps 1), KL at every site (eps 0.5, 1) | - | 3.5 |
| E9 | Nanbeige 2-loop completions: (a) kicks in both loops at once (`kick:...:apply=0,1`, same direction and size) at eps 0.5-1.5 + KL (`kick_kl.py --sites t0.l11+t1.l11`); (b) persona ablation: the trait elicited by the persona-vectors repo's system prompts, its vector projected out in loop 1, 2 or both (+ random-direction control; `persona_steer.py --mode ablate`); (c) refusal grid with norm-matched actadd at c = -1.5 and -2 in loop 1, 2, both (test set) | E2, E5 | 4.5 |
| E10 | Weight orthogonalization on the looped models at their trained loops: plain, norm-aware (`na=1`; Ouro and Dory, where sandwich-norm gains / the per-dimension update rate change it; Nanbeige is pre-norm, so plain is exact for its writers) and a random direction. Refusal on the test set (250; added to E3 `grid_test` for Ouro and Nanbeige, E4 `ortho_test` for Dory) and capability under tag `cap`: MATH-500 (500), AIME25 (30 x 8), MMLU-Pro (280: 20 per category, seed 0; with none and hook ablation) | E3, E4 | 15 |

Hours are for one 96 GB RTX PRO 6000; the judge (gpt-oss-20b) runs on the same GPU after each generation step.
Biggest single jobs: E3 nanbeige4 (7 h), E7 ouro4 (6 h), E3 dory4 (5 h).

Outputs (under `data/runs/<model>/`): E2 `generations/tune_actadd`, `capability/math500/tune_actadd`,
`analysis/actadd_tune.json`; E3 `generations/grid_test{,4,6}`, `analysis/grid_test*.csv`; E4 `generations/ortho_test{,4}`;
E5/E6 `persona/steer_{tune,loops,loops_loops4}*.parquet`, `analysis/persona_*.{csv,json}`; E7/E8
`capability/math500/kickp*`, `analysis/kick_kl/kickp*.parquet`; E9 `persona/steer_ablate*.parquet`, `analysis/persona_ablate.csv`, `analysis/kick_kl/kickp_both.parquet` (summary: `kick_summary.py --tag kickp --kl-tag kickp --kl-tag kickp_both`).

Reading results: `scripts/grid_ci.py data/runs/<model>/generations/<tag>`, `scripts/kick_summary.py --model <run dir>
--tag kickp`, `analysis/persona_loops*.csv`.

## Running on a new instance

1. Restore `/workspace` (and `/root`) from the source mirror as in `/workspace/INSTANCE_SETUP.md`. The checkout, its
   `.venv`, the model caches (`/workspace/cache/hf`) and `data/` (directions, persona vectors, stream norms, clean
   traces, earlier grids) all come with it. Keep the absolute paths; the venv needs an NVIDIA driver for CUDA 12.8+.
2. In every shell: `source /workspace/cloud-sync/env.sh` (gcloud, `GCS_BUCKET`, `GCLOUD_ACCOUNT`), then
   `cd /workspace/mats_projects/loop_steer && source scripts/env.sh`. Sign in once:
   `gcloud auth login "$GCLOUD_ACCOUNT" --no-launch-browser`.
3. Update the code: `git pull` (the mirror may predate the latest commit). Do not run `uv sync` unless `uv.lock`
   changed, and never while jobs run.
4. Check: `nvidia-smi`; `scripts/verify_setup.sh --gpu 0` (about 10 min; HF and vLLM hooks), or at least
   `.venv/bin/python scripts/exp.py list`.
5. Run: `scripts/exp.py run E7 --gpus 0,1` (longest jobs first, spread over the GPUs; a subset with
   `--jobs ouro4,qwen`). E3 and E6 first pull their dependencies (E2 / E5 outputs) from GCS and refuse to start
   without them.
6. Watch: `scripts/exp.py status E7`; logs in `data/runs/logs/gpuq/` and `data/runs/logs/`. A job silent for an hour
   is killed and retried (twice). Each job pushes its outputs to `$GCS_BUCKET/loop_steer/results/<EXP>/<host>/` when
   it ends, success or not; `scripts/exp.py push E7` re-pushes by hand.
7. Collect anywhere: `scripts/exp.py pull E2 E3 ...` copies every host's outputs into `data/runs` (never overwriting
   local files).

Rules: run each (experiment, job) on one host only (two hosts would race on the same files and GCS paths); jobs need
one whole GPU each (vLLM takes 85% of it); `LOOP_STEER_HOST` overrides the host name used in GCS paths; the queue
lives in `data/exp_queue/<host>` so a restored `data/queue` from another machine is never run.
