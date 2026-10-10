#!/usr/bin/env python3
"""A small FIFO job queue for one GPU: add commands, a runner executes them one after the other.

    scripts/gpuq.py add [-g GPU] [-n NAME] [--idle SEC] [--retries N] [--max-time SEC] [-e KEY=VAL] -- COMMAND ...
    scripts/gpuq.py start [-g GPU] [--hold-while CMD] [--free-mb MB]     # runner in the background (or `run` to stay in front)
    scripts/gpuq.py list [-g GPU]                                        # running / pending / recent done and failed
    scripts/gpuq.py cancel ID [--kill] | stop | log [ID]

Typical use: ``scripts/gpuq.py add -n ouro-aime -- .venv/bin/python scripts/capability.py --gpu '$GPU' ...`` then
``scripts/gpuq.py start`` once. Jobs run in the checkout directory with ``GPU`` (and ``GPUQ_GPU``) set to the queue's
GPU, so write ``--gpu $GPU`` (single-quote it when adding so your shell does not expand it) or use scripts that read
``${GPU}``. Command text is run by ``bash -c``, so ``;`` and ``&&`` chains work when quoted as one argument.

* Order is first in, first out; jobs can be added while the runner works. Logs: data/runs/logs/gpuq/<id>-<name>.log.
* Before each job the runner waits until the GPU holds less than ``--free-mb`` MiB (default 2000), so it does not collide
  with other users of a shared machine, and while ``--hold-while CMD`` succeeds, e.g. ``'pgrep -f "[r]un_followups.sh"'``
  (the bracket keeps pgrep from matching the shell that runs the check).
* Watchdog: a job whose log is silent for ``--idle`` seconds (default 1800) is killed with its whole process group and
  retried up to ``--retries`` times (default 2 extra attempts). Jobs should be skip-existing so a retry resumes. A job
  that still fails is moved to failed/ and the queue continues. ``--max-time`` caps one attempt's wall time.
* State lives under data/queue/gpu<N>/{pending,running,done,failed} (override the root with $GPUQ_ROOT): one JSON file per
  job, so nothing is lost if the runner dies; a restarted runner requeues a job whose process is gone.
Standard library only (no virtualenv needed).
"""

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QROOT = Path(os.environ.get("GPUQ_ROOT", ROOT / "data" / "queue"))
LOGS = Path(os.environ.get("GPUQ_LOGS", ROOT / "data" / "runs" / "logs" / "gpuq"))
RETRY_DELAY = int(os.environ.get("GPUQ_RETRY_DELAY", 30))
STATES = ("pending", "running", "done", "failed")


def qdir(gpu: str, state: str | None = None) -> Path:
    d = QROOT / f"gpu{gpu}" / (state or "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, job: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=1))
    tmp.rename(path)


def find_job(gpu: str, job_id: str):
    for state in STATES:
        for p in sorted(qdir(gpu, state).glob("*.json")):
            if p.stem == job_id or p.stem.endswith(job_id) or job_id in p.stem:
                return state, p
    return None, None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_used_mb(gpu: str):
    try:
        out = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().split()[0]) if out.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


# ------------------------------------------------------------------ commands
def cmd_add(a):
    cmd = " ".join(a.command) if len(a.command) == 1 else " ".join(shlex.quote(x) if "$GPU" not in x else x for x in a.command)
    if not cmd.strip():
        sys.exit("no command given (put it after --)")
    job_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid() % 1000:03d}"
    name = a.name or Path(a.command[0]).stem
    job = {"id": job_id, "name": name, "gpu": a.gpu, "cmd": cmd, "cwd": str(ROOT), "idle": a.idle, "retries": a.retries,
           "max_time": a.max_time, "env": dict(e.split("=", 1) for e in a.env), "added": now(), "attempts": []}
    write(qdir(a.gpu, "pending") / f"{job_id}-{name}.json", job)
    print(f"queued {job_id}-{name} on GPU {a.gpu}: {cmd}")


def cmd_list(a):
    for state in STATES:
        files = sorted(qdir(a.gpu, state).glob("*.json"))
        files = files[-8:] if state in ("done", "failed") else files
        for p in files:
            j = read(p)
            last = j["attempts"][-1] if j["attempts"] else {}
            extra = {"running": f"since {last.get('start', '?')} (attempt {len(j['attempts'])}, pid {last.get('pid')})",
                     "done": f"finished {last.get('end', '?')}", "failed": f"rc={last.get('rc')} {last.get('why', '')}",
                     "pending": f"added {j['added']}"}[state]
            print(f"{state:8s} {p.stem:48s} {extra}")
    print(f"(runner log: {qdir(a.gpu) / 'runner.log'})")


def cmd_cancel(a):
    state, p = find_job(a.gpu, a.id)
    if p is None:
        sys.exit(f"no job matching {a.id}")
    if state == "pending":
        p.unlink(); print(f"cancelled {p.stem}")
    elif state == "running" and a.kill:
        (qdir(a.gpu) / "cancel").write_text(p.stem); print(f"asked the runner to kill {p.stem}")
    else:
        sys.exit(f"{p.stem} is {state}" + ("; use --kill to stop a running job" if state == "running" else ""))


def cmd_stop(a):
    (qdir(a.gpu) / "stop").write_text(now()); print("runner will exit after the current job")


def cmd_log(a):
    files = sorted(LOGS.glob("*.log"))
    if a.id:
        files = [f for f in files if a.id in f.name]
    if not files:
        sys.exit("no logs")
    print(files[-1]); os.execvp("tail", ["tail", "-n", "30", str(files[-1])])


def cmd_start(a):
    log = open(qdir(a.gpu) / "runner.log", "a")
    args = [sys.executable, str(Path(__file__).resolve()), "run", "-g", a.gpu, "--free-mb", str(a.free_mb)]
    if a.hold_while:
        args += ["--hold-while", a.hold_while]
    p = subprocess.Popen(args, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT)
    print(f"runner for GPU {a.gpu} started (pid {p.pid}); log {qdir(a.gpu) / 'runner.log'}")


# ------------------------------------------------------------------ runner
def run_attempt(job: dict, log_path: Path, gpu: str, cancel_file: Path) -> dict:
    env = {**os.environ, "GPU": gpu, "GPUQ_GPU": gpu, **job["env"]}
    start = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n=== attempt {len(job['attempts']) + 1} {now()}: {job['cmd']}\n"); log.flush()
        proc = subprocess.Popen(["bash", "-c", job["cmd"]], cwd=job["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        att = {"start": now(), "pid": proc.pid}
        job["attempts"].append(att)
        write(job["_path"], {k: v for k, v in job.items() if k != "_path"})
        why = ""
        while proc.poll() is None:
            time.sleep(5)
            silent = time.time() - max(log_path.stat().st_mtime, start)
            if cancel_file.exists() and cancel_file.read_text().strip() == job["_path"].stem:
                cancel_file.unlink(); why = "cancelled"
            elif silent > job["idle"]:
                why = f"watchdog: no output for {int(silent)} s"
            elif job["max_time"] and time.time() - start > job["max_time"]:
                why = f"exceeded max-time {job['max_time']} s"
            if why:
                log.write(f"=== {why}; killing process group {proc.pid}\n"); log.flush()
                for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    t0 = time.time()
                    while proc.poll() is None and time.time() - t0 < wait:
                        time.sleep(1)
                proc.wait()
                break
        att.update(end=now(), rc=proc.returncode, why=why, seconds=int(time.time() - start))
    return att


def cmd_run(a):
    qdir(a.gpu); LOGS.mkdir(parents=True, exist_ok=True)
    lock = open(qdir(a.gpu) / "runner.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"a runner for GPU {a.gpu} is already running")
    stop, cancel = qdir(a.gpu) / "stop", qdir(a.gpu) / "cancel"
    stop.unlink(missing_ok=True); cancel.unlink(missing_ok=True)
    for p in sorted(qdir(a.gpu, "running").glob("*.json")):  # leftovers of a dead runner
        j = read(p)
        if not (j["attempts"] and alive(j["attempts"][-1].get("pid", 0))):
            p.rename(qdir(a.gpu, "pending") / p.name); print(f"[{now()}] requeued {p.stem} (its runner died)", flush=True)
    print(f"[{now()}] runner for GPU {a.gpu} up (free<{a.free_mb} MiB, hold-while={a.hold_while!r})", flush=True)
    idle_note = False
    while not stop.exists():
        pending = sorted(qdir(a.gpu, "pending").glob("*.json"))
        if not pending:
            if not idle_note:
                print(f"[{now()}] queue empty; waiting for jobs", flush=True); idle_note = True
            time.sleep(10); continue
        idle_note = False
        # wait until the GPU is free and nothing we must not overlap with is running
        while not stop.exists():
            used = gpu_used_mb(a.gpu)
            held = a.hold_while and subprocess.run(a.hold_while, shell=True, capture_output=True).returncode == 0
            if not held and (a.free_mb <= 0 or used is None or used < a.free_mb):
                break
            time.sleep(30)
        if stop.exists():
            break
        src = pending[0]
        if not src.exists():  # cancelled meanwhile
            continue
        path = qdir(a.gpu, "running") / src.name
        src.rename(path)
        job = read(path); job["_path"] = path
        log_path = LOGS / f"{path.stem}.log"
        print(f"[{now()}] start {path.stem}", flush=True)
        ok = False
        for attempt in range(job["retries"] + 1):
            att = run_attempt(job, log_path, a.gpu, cancel)
            print(f"[{now()}] {path.stem} attempt {attempt + 1}: rc={att['rc']} {att['why']} ({att['seconds']} s)", flush=True)
            if att["rc"] == 0:
                ok = True; break
            if att["why"] == "cancelled":
                break
            if attempt < job["retries"]:
                time.sleep(RETRY_DELAY)  # let the GPU memory be released before the retry
        write(path, {k: v for k, v in job.items() if k != "_path"})
        path.rename(qdir(a.gpu, "done" if ok else "failed") / path.name)
        print(f"[{now()}] {path.stem} -> {'done' if ok else 'failed'}", flush=True)
    print(f"[{now()}] runner stopped", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gpu = lambda p: p.add_argument("-g", "--gpu", default=os.environ.get("GPU", "0"))
    p = sub.add_parser("add"); gpu(p)
    p.add_argument("-n", "--name"); p.add_argument("--idle", type=int, default=1800)
    p.add_argument("--retries", type=int, default=2); p.add_argument("--max-time", type=int, default=0)
    p.add_argument("-e", "--env", action="append", default=[], help="extra KEY=VAL for the job")
    p.add_argument("command", nargs=argparse.REMAINDER); p.set_defaults(f=cmd_add)
    for name, f, extra in (("list", cmd_list, []), ("stop", cmd_stop, []), ("start", cmd_start, ["run"]), ("run", cmd_run, ["run"])):
        p = sub.add_parser(name); gpu(p); p.set_defaults(f=f)
        if extra:
            p.add_argument("--hold-while", default="", help="shell command; the runner waits while it succeeds")
            p.add_argument("--free-mb", type=int, default=2000, help="wait until the GPU uses less than this (0 = do not wait)")
    p = sub.add_parser("cancel"); gpu(p); p.add_argument("id"); p.add_argument("--kill", action="store_true"); p.set_defaults(f=cmd_cancel)
    p = sub.add_parser("log"); gpu(p); p.add_argument("id", nargs="?"); p.set_defaults(f=cmd_log)
    a = ap.parse_args()
    if a.cmd == "add" and a.command and a.command[0] == "--":
        a.command = a.command[1:]
    a.f(a)


if __name__ == "__main__":
    main()
