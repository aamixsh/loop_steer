"""Storage locations and job environment.

Run artifacts go to ``<data>/runs/<model>`` where ``<data>`` is ``$LOOP_STEER_DATA`` or the repo's
``data/`` directory (a symlink to bulk storage works well). Prompt CSVs are found via
``$LOOP_STEER_PROMPTS``, ``$DATA_DIR/datasets/reasoning-manipulation/prompts`` or ``data/prompts``.
Per-job temporary files go to a stamped directory under ``scratch/`` (also a symlink-friendly
location). On a shared server with a storage profile at ``/etc/profile.d/lab-storage.sh``,
``setup_job_env`` loads it and the profile decides where caches live. Elsewhere every cache (Hugging
Face, vLLM, Triton, torch, XDG) defaults to ``<data>/.cache``, so a checkout is self-contained.
"""

import atexit
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = "loop_steer"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("LOOP_STEER_DATA", REPO_ROOT / "data"))
SCRATCH_ROOT = REPO_ROOT / "scratch"
LAB_PROFILE = Path("/etc/profile.d/lab-storage.sh")


def _load_lab_profile() -> None:
    """Import the variables exported by the lab storage profile (existing values win)."""
    if not LAB_PROFILE.exists():
        return
    out = subprocess.run(["bash", "-c", f"source {LAB_PROFILE} && env -0"],
                         capture_output=True, check=True, env={"HOME": os.environ.get("HOME", ""),
                                                               "PATH": os.environ.get("PATH", "")}).stdout
    for item in out.split(b"\0"):
        key, sep, value = item.decode().partition("=")
        if sep and key not in ("PWD", "SHLVL", "_", "OLDPWD"):
            os.environ.setdefault(key, value)


def setup_job_env() -> Path:
    """Prepare a job: lab profile env, umask 077, vLLM cache on /data, stamped TMPDIR.

    Call before importing torch/transformers/vllm. Returns the job's temp dir
    (``scratch/<UTC>--loop_steer--<id>``), which child processes inherit via TMPDIR.
    """
    os.umask(0o077)
    _load_lab_profile()
    data_dir = _data_dir()
    if not LAB_PROFILE.exists():
        contain_caches(data_dir / ".cache")
    os.environ.setdefault("VLLM_CACHE_ROOT", str(data_dir / ".cache" / "vllm"))
    # Inductor defaults to $TMPDIR/torchinductor_$USER; vLLM's compile cache records that
    # absolute path, so a per-job TMPDIR would be re-created by later jobs. Keep it stable.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(data_dir / ".cache" / "torchinductor"))
    if os.environ.get("LOOP_STEER_JOB_TMP"):  # already set up (e.g. in a parent process)
        return Path(os.environ["LOOP_STEER_JOB_TMP"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = uuid.uuid4().hex[:8]
    tmp = SCRATCH_ROOT.resolve() / f"{stamp}--{PROJECT}--{job_id}"
    tmp.mkdir(parents=True)
    os.environ["TMPDIR"] = os.environ["LOOP_STEER_JOB_TMP"] = str(tmp)
    atexit.register(_prune_empty_dirs, tmp)
    _short_ipc_dir(job_id)
    return tmp


def _prune_empty_dirs(root: Path) -> None:
    """Remove ``root`` and its subdirectories if they hold no files (jobs often leave only empty cache dirs)."""
    for dirpath, _, _ in os.walk(root, topdown=False):
        try:
            os.rmdir(dirpath)
        except OSError:  # not empty: keep whatever the job left there
            pass


def _short_ipc_dir(job_id: str) -> None:
    """Give vLLM a socket directory whose paths fit the 107-character Unix socket limit.

    vLLM binds its IPC sockets as ``$VLLM_RPC_BASE_PATH/<uuid4>`` (36 characters) and defaults the base to
    ``$TMPDIR``, which the stamped job directory makes too long in a deeply nested checkout. Use a short
    per-job directory under ``scratch/`` and remove it (and the sockets vLLM leaves in it) at exit. If even that is
    too long, leave vLLM's default (the system temp dir). An explicit ``VLLM_RPC_BASE_PATH`` wins.
    """
    if "VLLM_RPC_BASE_PATH" in os.environ:
        return
    ipc = SCRATCH_ROOT.resolve() / f"ipc-{job_id}"
    if len(str(ipc)) + 1 + 36 > 107:
        return
    ipc.mkdir()
    os.environ["VLLM_RPC_BASE_PATH"] = str(ipc)
    atexit.register(shutil.rmtree, ipc, ignore_errors=True)


def contain_caches(cache: Path) -> None:
    """Point the tool caches under ``cache`` unless already set (existing values win)."""
    for key, sub in (("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "xdg"), ("TRITON_CACHE_DIR", "triton"),
                     ("TORCH_HOME", "torch")):
        os.environ.setdefault(key, str(cache / sub))


def _data_dir() -> Path:
    """Root for caches: $DATA_DIR if set (shared servers), else the repo's data/ directory."""
    return Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else DATA_ROOT


def dataset_dir() -> Path:
    """Upstream reasoning-manipulation prompt CSVs (``train_harmful_prompts.csv`` etc.)."""
    candidates = []
    if os.environ.get("LOOP_STEER_PROMPTS"):
        candidates.append(Path(os.environ["LOOP_STEER_PROMPTS"]))
    if os.environ.get("DATA_DIR"):
        candidates.append(Path(os.environ["DATA_DIR"]) / "datasets" / "reasoning-manipulation" / "prompts")
    candidates.append(DATA_ROOT / "prompts")
    return next((c for c in candidates if c.is_dir()), candidates[-1])


def run_dir(model_name: str) -> Path:
    """Per-model artifact directory, e.g. data/runs/Qwen3-8B."""
    path = DATA_ROOT / "runs" / model_name.split("/")[-1]
    path.mkdir(parents=True, exist_ok=True)
    return path
