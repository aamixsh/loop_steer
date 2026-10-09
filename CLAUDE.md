# loop-steer

Refusal-direction ablation and steering for looped reasoning models. Read `README.md` (setup, pipeline, metrics)
and `docs/handoff.md` (status, open items, pitfalls) first.

- Always use `.venv/bin/python` / `uv run`; the environment is locked (`uv sync --locked`). Do not sync or recreate
  `.venv` while jobs are running.
- On a machine without the lab storage profile, `source scripts/env.sh` first so every cache and temp file stays
  inside the checkout (`data/.cache`, `scratch/`).
- GPU scripts take `--gpu N` and call `loop_steer.paths.setup_job_env()`. Run artifacts live in `data/`, job temp files
  in `scratch/` (both may be symlinks to bulk storage).
- Do not touch `lab_envs/` (machine-specific, ignored by git).
- Keep the pinned model revisions (`src/loop_steer/models.py`, `judge.py`); results depend on them.
- Do not commit data, logs, credentials or model weights.
