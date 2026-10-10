# Setting up a new machine

Guide for bringing loop-steer up on a fresh GPU server (vast.ai, RunPod, a lab box) and picking up the work. Point the
agent or person on the new machine at this file. Current research status and the next steps are in
[PLAN.md](../PLAN.md); results are in the [README](../README.md), [presentation.md](presentation.md) and `results/`.

## 1. What the machine needs

| | |
|---|---|
| OS / tools | Linux, `git`, `curl`, `rsync`, `tar` (`zstd` optional: the bundle scripts fall back to gzip) |
| GPU | 1 NVIDIA GPU, 24 GB or more, driver supporting **CUDA 12.8+** (`nvidia-smi`). 2 GPUs let two jobs run in parallel. Developed on RTX PRO 6000 Blackwell, 96 GB |
| Disk | about **120 GB free**: models 40 GB, uv cache + venv 30 GB, run data 7 GB (+6 GB activations) |
| Python | none needed: uv installs the pinned 3.12 |

vLLM takes 85% of a GPU by default, so run **one job per GPU** (the judge, `gpt-oss-20b`, is loaded in-process by
`judge.py` and also needs a free GPU). Reference timings on one 96 GB GPU: Ouro selection candidate (50 prompts)
about 5 min, test-set candidate (487 prompts) 45 to 60 min, judging 3 to 5 min per candidate, Nanbeige MATH-500 about
15 min per candidate.

## 2. Bring-up (about 30 min, mostly downloads)

```bash
git clone https://github.com/aamixsh/loop_steer.git && cd loop_steer
# On vast.ai / RunPod keep the big caches on the persistent volume (/workspace), shared by all projects:
scripts/setup_remote.sh --install-uv --cache-dir /workspace/cache
# Without a persistent volume, drop --cache-dir: everything stays in data/.cache inside the checkout.
source scripts/env.sh        # in every new shell
scripts/verify_setup.sh --gpu 0
```

`setup_remote.sh` checks the driver and disk, installs uv, runs `uv sync --locked`, fetches the prompt CSVs at the
pinned upstream commit, downloads the four models at the pinned revisions and runs the CPU checks of
`verify_setup.sh`. Add the `export` lines it prints to `~/.bashrc` to make the shared caches the default.
`verify_setup.sh --gpu N` then runs the GPU smoke test: Hugging Face hooks on Ouro and Nanbeige and vLLM hooks against
Hugging Face. Fix any FAIL before starting jobs.

## 3. Previous runs (optional)

Summary tables of all runs are in git under `results/`, so reading results needs no data. To continue experiments that
reuse saved directions or generations, copy the data bundle:

```bash
# old machine
scripts/pack_data.sh bundle.tar.zst                  # directions, generations, analysis, logs, prompts (about 0.6 GB)
scripts/pack_data.sh bundle.tar.zst --with-activations   # +6.3 GB; only needed to recompute directions
rsync -avP bundle.tar.zst bundle.tar.zst.sha256 USER@NEWHOST:loop_steer/
# new machine
scripts/unpack_data.sh bundle.tar.zst                # checks the checksum, never overwrites existing files
scripts/verify_setup.sh --no-gpu                     # re-summarizes a saved run and compares with the README
```

Do **not** copy SSH keys or Hugging Face tokens. All models and the prompts are public (GPQA, if ever used, is gated
and needs your own token and license acceptance). After a copy, `scripts/export_results.sh` refreshes `results/`.

## 4. Where things live

- Code in git; **never commit** `data/`, logs, credentials or weights (`.gitignore` covers them).
- `data/runs/<model>/{activations,directions,generations,analysis,figures,capability}`, logs in `data/runs/logs`.
- Caches: `data/.cache/{huggingface,uv,uv-python}` are symlinks to the shared cache when `--cache-dir` was used;
  Triton, torch, vLLM and Inductor caches stay in `data/.cache`. Job temp files: `scratch/`.
- Every GPU script takes `--gpu N` (batch chains: `GPU=N`), calls `loop_steer.paths.setup_job_env()` and is
  skip-existing, so re-running resumes. Batch chains and queues: `scripts/run_*.sh`, `scripts/queue_*.sh`.

## 5. Pitfalls (each one cost time once)

- **Jobs from non-interactive shells** (`nohup`, cron, agent tools) do not read `~/.bashrc`, so `HF_HOME` and friends
  are unset. The `data/.cache` symlinks and `source scripts/env.sh` make that harmless; without them a job silently
  downloads 40 GB into the checkout.
- **vLLM IPC path length.** vLLM binds sockets at `$VLLM_RPC_BASE_PATH/<uuid>`, a Unix socket path limited to 107
  characters. `setup_job_env()` uses a short `scratch/ipc-<id>`; scripts must call it before importing vLLM. A deeply
  nested checkout otherwise fails with "ipc path ... is longer than 107 characters" (`verify_setup.sh` checks this).
- **Do not run `uv sync` or recreate `.venv` while jobs run**, and do not edit a bash queue script while it runs (bash
  reads it incrementally); put per-step changes in a script the queue calls fresh for each step.
- **Hooks inside vLLM** need `VLLM_ALLOW_INSECURE_SERIALIZATION=1` and eager mode; Nanbeige needs its in-process port
  (`vllm_nanbeige.py`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`). The scripts set both.
- **Pinned revisions** of Qwen3-8B, Ouro, Nanbeige and the judge (`src/loop_steer/models.py`, `judge.py`) produce the
  reported numbers; do not change them. Pins: torch 2.10.0+cu128, vLLM 0.18.0, transformers 4.57.6.
- `validate_vllm_hooks.py` defaults to `--vllm-mem 0.1`, too small for the 3B and 8B models; use 0.3.
- `summarize.py --max-tokens` must equal the stage-1 cap of the run (2048 for Qwen and Ouro, 4096 for Nanbeige).
- `ortho.orthogonalize_` must keep the keyword `direction` (`intervene.py` calls it by keyword).
- Sampling is temperature 0.6, top_p 1.0, no top_k for every model.
- Offline runs: `HF_HUB_OFFLINE=1` stops accidental downloads once the models are cached.

## 6. Carrying the agent's context (optional)

`PLAN.md` holds the plan and next steps, so a new agent needs nothing else. A Claude Code conversation and its memory
can also be copied to `~/.claude/projects/<cwd with every non-alphanumeric character replaced by "-">/` (seen for `/` and `_`).
