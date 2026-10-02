"""Storage locations and job environment (lab layout, see ~/STARTUP.md).

Code lives in the repo (backed-up HOME). Run artifacts go through the repo's
``data`` link (-> /data/$USER/projects/loop_steer); datasets live in the
user-global $DATA_DIR/datasets; per-job temporary files go to a
timestamp/project-stamped directory under the ``scratch`` link
(-> /scr/$USER/tmp). Model weights come from the shared Hugging Face cache.
"""

import os
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
    data_dir = Path(os.environ.get("DATA_DIR", f"/data/{os.environ.get('USER', '')}"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(data_dir / ".cache" / "vllm"))
    # Inductor defaults to $TMPDIR/torchinductor_$USER; vLLM's compile cache records that
    # absolute path, so a per-job TMPDIR would be re-created by later jobs. Keep it stable.
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(data_dir / ".cache" / "torchinductor"))
    if os.environ.get("LOOP_STEER_JOB_TMP"):  # already set up (e.g. in a parent process)
        return Path(os.environ["LOOP_STEER_JOB_TMP"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp = SCRATCH_ROOT.resolve() / f"{stamp}--{PROJECT}--{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    os.environ["TMPDIR"] = os.environ["LOOP_STEER_JOB_TMP"] = str(tmp)
    return tmp


def dataset_dir() -> Path:
    """Upstream reasoning-manipulation prompt CSVs (user-global datasets/)."""
    data_dir = Path(os.environ.get("DATA_DIR", f"/data/{os.environ.get('USER', '')}"))
    return data_dir / "datasets" / "reasoning-manipulation" / "prompts"


def run_dir(model_name: str) -> Path:
    """Per-model artifact directory, e.g. data/runs/Qwen3-8B."""
    path = DATA_ROOT / "runs" / model_name.split("/")[-1]
    path.mkdir(parents=True, exist_ok=True)
    return path
