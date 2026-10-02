# loop-steer

Activation steering experiments with [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking).

New to this system? Start with [Setup 101: shared caches, uv, and Hugging Face](SETUP_101.md).

## Start the notebook

```bash
uv sync --locked
uv run python -m ipykernel install --prefix .venv --name loop-steer --display-name 'Python (loop-steer)'
uv run jupyter lab notebooks/01_streaming.ipynb
```

Choose the **Python (loop-steer)** kernel and run the notebook from the top. Set
`GPU_ID` in the first code cell to `"0"` or `"1"` **before importing torch**.
Restart the kernel before changing GPUs. Only the selected GPU is visible to
the notebook, where it becomes `cuda:0`. The notebook streams a response using
Transformers and leaves `model` and `tokenizer` available for activation hooks.

The kernel is already registered in this checkout. Repeat its installation
command after recreating `.venv`.

## Terminal environment and Hugging Face login

Activate the environment in a Bash or Zsh terminal:

```bash
cd /home/amishr24/loop_steer
source .venv/bin/activate
```

`python`, `hf`, `huggingface-cli`, `jupyter`, and `vllm` then resolve to the
project environment. Run `deactivate` to leave it. Activation is specific to
each terminal. Alternatively, prefix a command with `uv run` from this project,
for example `uv run hf auth login`.

To save your Hugging Face token, run the interactive login and paste it at the
prompt:

```bash
hf auth login
hf auth whoami
```

The installed Hub version also supports `huggingface-cli login`; `hf auth login`
is the current command. This machine sets `HF_HOME` to
`/home/amishr24/.cache/huggingface` and `HF_TOKEN_PATH` to
`/home/amishr24/.cache/huggingface/token`. Login therefore saves credentials
in your home directory, while `HF_HUB_CACHE` keeps model downloads in the
shared cache. Other environments using the same token path can reuse the login.

## vLLM

Start the local OpenAI-compatible server in another terminal:

```bash
uv run python scripts/serve_vllm.py --gpu 1
```

The notebook has an optional cell for streaming from this server. It defaults
to `http://127.0.0.1:8000/v1`. To use GPU 0 for vLLM, release the notebook model
with its cleanup cell first. Stop the server with Ctrl-C.

The launcher selects one GPU, binds to localhost, uses BF16, limits the context
to 4096 tokens, reserves 20% of that GPU's memory, and disables graph compilation
for quick research startup. Override these defaults with vLLM options, e.g.:

```bash
uv run python scripts/serve_vllm.py --gpu 1 --max-model-len 8192 --gpu-memory-utilization 0.3
```

vLLM uses its native `OuroForCausalLM` implementation. It executes all recurrent
steps; adaptive exit is supported only on the Transformers path. Notebook hooks
operate on the in-process Transformers model; vLLM serves a separate model.

## Reproducibility and storage

uv-managed Python 3.12.14, PyTorch 2.10.0, Transformers 4.57.6, and vLLM 0.18.0
are pinned, and `uv.lock` records all resolved dependencies. This stack keeps Transformers
4 compatibility for Ouro's custom code while including native vLLM support and
CUDA wheels for Blackwell. The model weights and custom code are pinned to
revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, including the upstream KV
cache fix. Loading this revision requires `trust_remote_code=True`.

The managed Python includes development headers needed by Triton's runtime
compiler. The system Python on this machine lacks these headers; `uv sync`
automatically uses the pinned managed interpreter instead.

The model has 24 shared decoder layers and, by default, four recurrent passes.
A hook on `model.model.layers[i]` therefore fires four times per forward pass.
Use `register_forward_hook(..., with_kwargs=True)` and the `current_ut` keyword
to identify the zero-based pass index when steering a specific recurrence. Keep
hook handles and call `handle.remove()` when an experiment ends.

Cache directories follow the inherited environment settings and library
defaults. On this machine:

| Purpose | Location |
| --- | --- |
| uv packages (`UV_CACHE_DIR`) | `/data/shared_lab/uv_cache` |
| Hugging Face models (`HF_HUB_CACHE`) | `/data/shared_lab/hf_cache/hub` |
| Hugging Face home (`HF_HOME`) | `/home/amishr24/.cache/huggingface` |
| Active Hugging Face token (`HF_TOKEN_PATH`) | `/home/amishr24/.cache/huggingface/token` |
| vLLM cache (`VLLM_CACHE_ROOT`, set by `scripts/serve_vllm.py` and `setup_job_env()`) | `/data/amishr24/.cache/vllm` |
| Triton cache (library default) | `~/.triton/cache` |

The notebook and server launcher inherit these settings. Restart existing
notebook kernels after changing cache environment variables, since libraries
read them during import. Model weights, generated notebook outputs, experiment
data, and checkpoints should not be committed. Clear notebook outputs before
committing notebooks.

Validated on this machine's RTX PRO 6000 Blackwell GPUs: notebook inference on
GPU 0, activation hooks across all four recurrent passes, propagation of
generation errors, and vLLM chat streaming on GPU 1. Local validation outputs
are saved under `outputs/validation/`.

Shared-cache validation also runs both streaming paths with Hugging Face
downloads disabled after deleting the project `.cache/` directory. It confirms
that model files come from the shared cache and the project cache is not
recreated.

References: [model source](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/blob/main/modeling_ouro.py),
[vLLM supported models](https://docs.vllm.ai/en/v0.18.0/models/supported_models/).

## Refusal-direction pipeline

Reproduces the reasoning-model refusal-direction method of
[reasoning-manipulation](https://github.com/kureha-yamaguchi/reasoning-manipulation)
(arXiv 2507.03167, built on [refusal_direction](https://github.com/andyrdt/refusal_direction))
on Qwen3-8B, then ports it to Ouro.

Storage follows the lab layout in `~/STARTUP.md`. Run artifacts go to
`data/runs/<model>/` (the `data` link points to `/data/$USER/projects/loop_steer`;
override with `LOOP_STEER_DATA`) and logs to `data/runs/logs/`. Prompts are the upstream
train/test CSVs in `$DATA_DIR/datasets/reasoning-manipulation/prompts/`, next to the
upstream Qwen3-8B reference directions (`reference-qwen3-8b/`). GPU scripts call
`loop_steer.paths.setup_job_env()`, which loads `/etc/profile.d/lab-storage.sh`, sets
umask 077, puts the vLLM and TorchInductor caches under `$DATA_DIR/.cache/`, and points `TMPDIR` at a
fresh `scratch/<UTC>--loop_steer--<id>` directory (`scratch` points to `/scr/$USER/tmp`).
Delete those job directories once the job is finished.

| Step | Script | Output |
| --- | --- | --- |
| Optional judge server (gpt-oss-20b, port 8001; use with `judge.py --url`) | `scripts/serve_judge.py --gpu 1` | — |
| Two-stage sampling: CoTs, then answers per CoT | `scripts/generate.py` | `generations/*.parquet` |
| StrongREJECT rubric scores (bounded batch job, judge loaded in-process) | `scripts/judge.py` | `*.scored.parquet` |
| Refusal / non-refusal labels (CoT- and prompt-level) | `scripts/build_sets.py` | `generations/labels.parquet` |
| Window-mean resid_pre at every (loop, layer) | `scripts/extract_activations.py` | `activations/<name>.pt` |
| Difference-in-means directions | `scripts/directions.py` | `directions/<name>.pt` |
| Per-site stability / separability | `scripts/analyze_directions.py` | `analysis/`, `figures/` |
| Generation under interventions | `scripts/intervene.py` | `generations/<tag>/` |
| Summary table | `scripts/summarize.py` | — |

Direction methods: `v4_cot` (mean over all CoT tokens; upstream v4), `v12_cot150` (first
150 CoT tokens; upstream v1/v2), `cot_last150`, `v4_baseline` (end-of-instruction tokens,
prompt-level labels), `paired_cot` (within-prompt refusal-minus-compliance contrast).

Interventions (`scripts/intervene.py --candidates kind:method:site[:opts]`):
`ortho` projects the direction out of the residual-writing weights (equivalent to
ablation for pre-norm models; validated by `scripts/validate_ortho.py`), `ablate` does
the same with hooks, `actadd` adds a scaled direction at one layer. Hook interventions
run inside vLLM in eager mode (`loop_steer.vllm_hooks`, validated against the HF hooks
in `loop_steer.hooks` by `scripts/validate_vllm_hooks.py`).

**Ouro differences.** Sites are (loop, layer), 4 x 24. Ouro writes
`RMSNorm_2(sublayer(x))` into the residual and renormalizes between loops, so weight
orthogonalization does not remove a direction (`scripts/ouro_smoke.py` shows the
projection is larger after orthogonalization than before); use `ablate`. Hooks read
the loop index from `current_ut`. `output_hidden_states=True` raises in the pinned
remote code; activations come from hooks.
