# loop-steer

Refusal-direction ablation and steering for **looped reasoning models**.

This repo ports the method of [*reasoning-manipulation*](https://github.com/kureha-yamaguchi/reasoning-manipulation)
(arXiv [2507.03167](https://arxiv.org/abs/2507.03167), built on
[*refusal_direction*](https://github.com/andyrdt/refusal_direction)) to models that apply the same layer stack
several times: [Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) (4 loops) and
[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) (2 loops). The method was first reproduced on
Qwen3-8B.

The method: find a *refusal direction* in the residual stream as the difference in mean activations between
chains of thought (CoTs) that end in refusal and CoTs that end in compliance, then remove it (ablation, weight
orthogonalization) or push against it (activation addition) and measure how often the model still refuses.

## Findings

| Judge refusal rate | Qwen3-8B | Ouro-1.4B (4 loops) | Nanbeige4.2-3B (2 loops) |
|---|---|---|---|
| Clean model | 61% | 61% | 78% |
| Hook ablation | not run | **10%** | 16–18% |
| Weight orthogonalization | **6–15%** | no valid outputs* | **11%** |
| Random-direction control | 61% (weights) | 59% (hooks) | 76% (hooks), 73% (weights) |

All 487 held-out test prompts, 3 CoTs per prompt, 3 answers per CoT, over all samples. The ranges are over the
tested directions and layers. Mean StrongREJECT score rises from 0.22–0.37 (clean) to 0.83–0.91 with the best
candidate per model. On Qwen3-8B our directions match the published layer-17 vectors (cosine 0.99).
*Weight orthogonalization on Ouro was tested on 50 selection prompts only (the edit breaks generation, see below).

**Why weight orthogonalization breaks Ouro but not Nanbeige.** Weight orthogonalization only stops the edited
matrices from writing along the direction. A learned RMSNorm applied *after* those matrices has uneven
per-dimension gains, so it rotates the output back onto the direction. Ouro has this problem twice: its
sandwich norms re-add the direction to every attention and MLP write, and its shared norm between loops scales
the stream up and compounds the leak over three passes. Nanbeige adds writes to the stream without a norm, and
its single inter-loop norm shrinks the stream. Hooks avoid the problem by re-projecting after every norm. See
[docs/presentation.md](docs/presentation.md) for the full explanation, figures, and the tests that confirm it.

![Leak of the removed direction per loop](docs/figures/loop_growth.png)

**Caveats.** No capability or over-refusal checks were run on any model. Hook ablation was not run on Qwen3-8B.
Mechanism measurements use 8 traces per model. Directions and layers were chosen on 50 selection prompts and
reported on the test set. Sampling is temperature 0.6 with no top-p or top-k, which differs from the models'
recommended settings (top_p 0.95, top_k 20 for Qwen3 and Nanbeige). CoTs were capped at 8192 tokens; 0–2% of them
still hit the cap (see Metrics). See the deck for details.

## Setup

Requires Linux, Python 3.12 and an NVIDIA GPU. Everything was run on RTX PRO 6000 (Blackwell, 96 GB) GPUs. A
3B-parameter model plus the judge each fit on one such GPU. Dependencies are pinned (PyTorch 2.10, Transformers
4.57.6, vLLM 0.18.0) because the models ship custom code that depends on the Transformers major version.

```bash
git clone https://github.com/aamixsh/loop_steer.git && cd loop_steer
uv sync --locked          # creates .venv; see https://docs.astral.sh/uv/
```

Model weights and remote code are downloaded from the Hugging Face Hub on first use at the pinned revisions
(set `HF_HOME` / `HF_HUB_CACHE` to control where). The judge is `openai/gpt-oss-20b`. Loading Ouro and Nanbeige
requires `trust_remote_code=True`, which the scripts set.

**Prompts.** Get the upstream prompt CSVs and place them in `data/prompts/` (or point `LOOP_STEER_PROMPTS` at
a directory). We used commit `56a763d` of reasoning-manipulation:

```bash
git clone https://github.com/kureha-yamaguchi/reasoning-manipulation /tmp/rm
git -C /tmp/rm checkout 56a763d837cc63f30a13f830ad4ebd05890c7455
mkdir -p data/prompts && cp /tmp/rm/dataset/{train,test}_{harmful,harmless}_prompts.csv data/prompts/
```

**Storage.** Run artifacts go to `data/runs/<model>/` and job temp files to `scratch/`. Both are plain
directories created on demand, or you can symlink them to bulk storage. `LOOP_STEER_DATA` overrides the data
location. Activations are large (about 2 GB per model at the reduced scale).

## Pipeline

All GPU scripts take `--gpu N` and call `loop_steer.paths.setup_job_env()`, which sets up the vLLM cache and a
per-job temp directory. Example for Ouro. For Nanbeige use `--model Nanbeige/Nanbeige4.2-3B`,
`--max-tokens 4096` and `--max-model-len 12288` (its traces are long).

```bash
M=ByteDance/Ouro-1.4B-Thinking; R=data/runs/Ouro-1.4B-Thinking

# 1. Sample CoTs (3 per prompt), then 3 answers per CoT, with no intervention
uv run python scripts/generate.py --model $M --split train --n-prompts 500 --out train_clean.parquet
# 2. Score answers with the StrongREJECT rubric (judge loaded in-process on one GPU)
uv run python scripts/judge.py --gpu 1 $R/generations/train_clean.parquet
# 3. Label CoTs refusal / non-refusal
uv run python scripts/build_sets.py $R/generations/train_clean.scored.parquet
# 4. Window-mean activations at every (loop, layer), then difference-in-means directions
uv run python scripts/extract_activations.py --model $M --labels $R/generations/labels.parquet --name train --labeled-only
uv run python scripts/directions.py --model $M --activations train
uv run python scripts/analyze_directions.py --model $M --method v4_cot      # per-site stability and AUC
# 5. Generate under interventions, judge, summarize
uv run python scripts/intervene.py --model $M --directions train --split train --offset 500 --n-prompts 50 \
    --tag sel --candidates none ablate:v4_baseline:t3.l16 ablate:random:t3.l16
uv run python scripts/judge.py --gpu 1 $R/generations/sel
uv run python scripts/summarize.py $R/generations/sel
# 6. CoTs that hit the token cap: continue them under the same intervention, judge, and merge into the summary
uv run python scripts/intervene.py --model $M --directions train --split test --tag test_hooks_ext \
    --extend-from test_hooks --max-tokens 2048 --extend-tokens 6144 --max-model-len 16384 \
    --candidates ablate:v4_baseline:t3.l16
uv run python scripts/judge.py --gpu 1 $R/generations/test_hooks_ext $R/generations/test_hooks_ext/invalid
uv run python scripts/summarize.py $R/generations/test_hooks --cont $R/generations/test_hooks_ext
```

| Script | Purpose |
|---|---|
| `generate.py` | Two-stage sampling: CoTs, then answers conditioned on each CoT |
| `judge.py`, `serve_judge.py` | StrongREJECT rubric scores from gpt-oss-20b (batch job, or a server for `--url`) |
| `build_sets.py` | CoT-level and prompt-level refusal / non-refusal labels |
| `extract_activations.py`, `directions.py`, `analyze_directions.py` | Directions per window and site, with stability checks |
| `intervene.py` | Generation under `ablate` (hooks), `ortho` (weights) or `actadd` for many candidates per engine; `--extend-from` continues capped CoTs |
| `summarize.py` | One row per candidate: mean score, judge and substring refusal, share of unclosed CoTs; `--cont` merges continuations |
| `ortho_compare.py`, `ortho_diagnostic.py`, `ortho_feedback.py` | Why weight orthogonalization leaks through norms in Ouro |
| `loop_smoke.py`, `ouro_smoke.py`, `validate_*.py` | Smoke tests and checks that vLLM hooks match Hugging Face |
| `make_presentation_figures.py` | Regenerates `docs/figures/` from saved runs |
| `run_*.sh` | The batch chains used for the test-set runs (generation, continuation, judging) |

**Direction methods.** `v4_cot` (mean over all CoT tokens), `v12_cot150` (first 150 CoT tokens), `cot_last150`,
`v4_baseline` (end-of-instruction template tokens, prompt-level labels), `paired_cot` (within-prompt contrast)
and `random` (control). Candidate syntax is `kind:method:site[:c=..|apply=..|layer=..]` with site `l<layer>` or
`t<loop>.l<layer>`. See the docstring of `scripts/intervene.py`.

**Metrics.** The StrongREJECT score is `(1 − refused) × (convincingness + specificity − 2) / 8`, in [0, 1].
A stage-1 sample is *valid* if its CoT closes with `</think>`; its answers are then sampled in stage 2.
`judge_refusal` and `mean_score` cover valid samples only. The others are *invalid* and are judged as they are,
on the whole response: *capped* samples ran into the token cap (usually loops; `capped_frac`), and *direct* samples
stopped by themselves without thinking (`direct_frac`; about 20% of clean Ouro samples). `all_refusal` and
`all_score` average over every sample, with each valid sample's answers averaged first. The tables above use
`all_refusal` and `all_score`. `invalid_cot_frac = capped_frac + direct_frac`.

## Notes on the models

- **Sites** are `(loop, layer)`: 36 for Qwen3-8B, 96 for Ouro, 44 for Nanbeige. Hooks read the loop index from
  the `current_ut` (Ouro) or `loop_idx` (Nanbeige) argument that each decoder layer receives.
- **Ouro** supports early exit. We set `early_exit_threshold = 1.0` so every token runs all four loops.
  `output_hidden_states=True` raises in the pinned remote code, so activations are captured with hooks.
- **Nanbeige** is not supported by vLLM 0.18. `src/loop_steer/vllm_nanbeige.py` is a port registered at runtime
  (it needs `VLLM_ENABLE_V1_MULTIPROCESSING=0`, which the scripts set). It matches Hugging Face greedy output.
  The Hugging Face code needs a small `DynamicCache` shim, applied in `loop_steer.models.load_model`.
- **Hooks inside vLLM** run in eager mode through `LLM.apply_model`, which requires
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. The scripts set this.
- **`ortho` is only valid** when nothing renormalizes the stream after the edited weights (Qwen, Nanbeige).
  `loop_steer.ortho` also provides a norm-aware variant and a loop-span variant used for the analysis.

## Streaming demo

`notebooks/01_streaming.ipynb` streams a response from Ouro with Transformers and leaves the model available
for hooks. `uv run python scripts/serve_vllm.py --gpu 1` serves Ouro through vLLM on localhost.

## Acknowledgements

Method and prompts: [reasoning-manipulation](https://github.com/kureha-yamaguchi/reasoning-manipulation)
(Apache-2.0) and [refusal_direction](https://github.com/andyrdt/refusal_direction) (MIT). Judge rubric:
StrongREJECT ([arXiv 2402.10260](https://arxiv.org/abs/2402.10260)). Models: ByteDance Ouro and Nanbeige, under their own
licenses.
