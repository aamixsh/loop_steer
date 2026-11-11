#!/usr/bin/env python3
"""Numbered experiments (docs/experiments.md): queue their jobs on this machine's GPUs, sync results through GCS.

    scripts/exp.py list                         # experiments, jobs, estimated GPU hours, dependencies
    scripts/exp.py show E3                      # the exact commands of every job
    scripts/exp.py run E3 --gpus 0,1 [--jobs ouro4,qwen]   # pull dependencies, queue the jobs, start the runners
    scripts/exp.py status [E3]                  # queue state of the experiment's jobs on this machine
    scripts/exp.py push E3 [--job ouro4]        # upload the outputs of finished jobs (each job also does this itself)
    scripts/exp.py pull E3 [E5 ...]             # download every machine's outputs of these experiments into data/runs
    scripts/exp.py coef ouro                    # the tuned actadd coefficient (E2), as used by E3 jobs

Jobs run through ``scripts/gpuq.py`` (FIFO per GPU, free-GPU wait, idle watchdog, retries) with their own queue root
``data/exp_queue/<host>`` so queue state copied from another machine is never picked up. Each job is a shell command
(skip-existing, so a retry or a rerun resumes) that ends by pushing its outputs. Results live under
``$LOOP_STEER_GCS/results/<EXP>/<host>/`` (default ``$GCS_BUCKET/loop_steer``), mirroring paths below ``data/runs``;
one writer (host) per prefix, and ``pull`` never overwrites local files. GCS needs ``gcloud`` signed in and
``GCS_BUCKET`` / ``GCLOUD_ACCOUNT`` set (``source /workspace/cloud-sync/env.sh``, see /workspace/INSTANCE_SETUP.md).
Without them ``run`` still works and ``push`` is skipped with a warning (results stay local). Standard library only.
"""

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
HOST = os.environ.get("LOOP_STEER_HOST") or socket.gethostname()
QROOT = ROOT / "data" / "exp_queue" / HOST
PY = ".venv/bin/python"

M = {  # key: (HF id, run dir, refusal direction method:site, persona / kick mid-loop layer)
    "ouro": ("ByteDance/Ouro-1.4B-Thinking", "Ouro-1.4B-Thinking", "v4_baseline:t3.l16", 12),
    "nanbeige": ("Nanbeige/Nanbeige4.2-3B", "Nanbeige4.2-3B", "v4_baseline:t1.l15", 11),
    "dory": ("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "hh_eoi:t0.l26", 36),
    "qwen": ("Qwen/Qwen3-8B", "Qwen3-8B", "v4_baseline:l21", None),
}
QWEN_QUARTERS = {1: "0-8", 2: "9-17", 3: "18-26", 4: "27-35"}  # ablation layer ranges
QWEN_AT = {1: 3, 2: 12, 3: 21, 4: 30}  # actadd: the site's offset (21 = 2 * 9 + 3) in each quarter
TEST = "SPLIT=test OFFSET=0 NPROMPTS=250"


def apply_opt(loops):
    return "" if loops is None else ":apply=" + ",".join(map(str, loops))


def refusal_cands(key, sets, actadd=True):
    """none, random control, ablation and (tuned, norm-matched) actadd in each loop set (None = every loop)."""
    method, site = M[key][2].split(":")
    out = ["none", f"ablate:random:{site}"]
    out += [f"ablate:{method}:{site}{apply_opt(s)}" for s in sets]
    if actadd:
        out += [f"actadd:{method}:{site}:c=$COEF:rel=1{apply_opt(s)}" for s in sets]
    return out


def qwen_refusal_cands():
    quarters = [[1], [2], [3], [4], [1, 2], [1, 2, 3, 4]]
    out = ["none", "ablate:random:l21", "ablate:v4_baseline:l21"]  # plain ablation = every layer
    for q in quarters[:-1] + [[1, 2], [3, 4]]:  # quarters, first half, second half
        out.append("ablate:v4_baseline:l21:layers=" + "+".join(QWEN_QUARTERS[i] for i in q))
    for at in [[1], [2], [3], [4], [1, 2], [1, 2, 3, 4], [1, 3]]:  # [1, 3]: the halves' offsets (layers 3 and 21)
        out.append("actadd:v4_baseline:l21:c=$COEF:rel=1:at=" + "+".join(str(QWEN_AT[i]) for i in at))
    return list(dict.fromkeys(out))


def select(key, tag, cands, n_loops=None, test=True):
    env = (TEST + " " if test else "") + (f"N_LOOPS={n_loops} " if n_loops else "")
    coef = f"COEF=$({PY} scripts/exp.py coef {key}) && " if any("$COEF" in c for c in cands) else ""
    return coef + f"GPU=$GPU {env}scripts/run_select.sh {key} {tag} " + " ".join(shlex.quote(c) if "$" not in c
                                                                                 else f'"{c}"' for c in cands)


def kick_job(key, tag, sites, n_loops=None, eps="0.5,0.7,1,1.5", limit=100, kl_sites=None, kl_eps=None):
    hf = M[key][0]
    nl = f" --n-loops {n_loops}" if n_loops else ""
    cands = ["none"] + [f"kick:randp0:{s}:eps={e}" for s in sites for e in eps.split(",")]
    gen = (f"{PY} scripts/capability.py --gpu $GPU --model {hf} --dataset math500 --tag {tag} --limit {limit}{nl} "
           f"--skip-existing --candidates " + " ".join(cands))
    kl = (f"{PY} scripts/kick_kl.py --gpu $GPU --model {hf}{nl} --tag {tag} --limit {limit} --skip-existing "
          f"--sites {kl_sites or ','.join(sites)} --eps {kl_eps or eps}")
    return gen + " && " + kl


def tune_job(key, coeffs):
    hf, run, ms, _ = M[key]
    method, site = ms.split(":")
    extra = ":at=3+12+21+30" if key == "qwen" else ""
    cands = ["none"] + [f"actadd:{method}:{site}:c={c}:rel=1{extra}" for c in coeffs]
    sel = f"GPU=$GPU scripts/run_select.sh {key} tune_actadd " + " ".join(cands)
    cap = (f"{PY} scripts/capability.py --gpu $GPU --model {hf} --dataset math500 --tag tune_actadd --limit 100 "
           f"--skip-existing --candidates " + " ".join(cands))
    return f"{sel} && {cap} && {PY} scripts/actadd_tune.py --model {key}"


def persona(key, mode, sets=None, n_loops=None):
    nl = f" --n-loops {n_loops}" if n_loops else ""
    s = (" --sets " + " ".join(sets)) if sets else ""
    return f"{PY} scripts/persona_steer.py --gpu $GPU --model {M[key][0]} --mode {mode}{nl}{s}"


def out(key, *paths):
    return [f"{M[key][1]}/{p}" for p in paths]


ORTHO = ["none", "ablate:hh_eoi:t0.l26", "ortho:hh_eoi:t0.l26", "ortho:hh_eoi:t0.l26:na=1"]  # na=1: lr-aware edit
L4 = [[0], [1], [2], [3], [0, 1], None]  # 4 loops: single loops, first two, every loop
L2 = [[0], [1], None]
DORY_PILOT_SITES = (["t0.l4", "t0.l12"] + [f"t{t}.l{l}" for t in range(4) for l in range(16, 56, 4)]
                    + ["t0.l60", "t0.l68"])

EXPERIMENTS = {
    "E1": {"title": "Stream norms at every site (for norm-matched actadd, rel=1)", "needs": [], "jobs": {
        "norms": {"hours": 0.3, "cmd": " && ".join(
            f"{PY} scripts/stream_norms.py --gpu $GPU --model {M[k][0]} --skip-existing" + (f" --n-loops {n}" if n else "")
            for k, n in [("qwen", None), ("ouro", None), ("ouro", 6), ("nanbeige", None), ("nanbeige", 4),
                         ("dory", None), ("dory", 4)]),
            "outputs": [f"{M[k][1]}/directions" for k in M]},
    }},
    "E2": {"title": "Actadd coefficient tuning: refusal (selection, 50 prompts) + MATH-500 (100) per c, every loop",
           "needs": ["E1"], "jobs": {
        "ouro": {"hours": 2.5, "cmd": tune_job("ouro", [-0.5, -1, -1.5, -2])},
        "nanbeige": {"hours": 3.0, "cmd": tune_job("nanbeige", [-1, -1.5, -2, -3])},
        "dory": {"hours": 3.0, "cmd": tune_job("dory", [-0.5, -1, -1.5, -2])},
        "qwen": {"hours": 1.5, "cmd": tune_job("qwen", [-0.5, -1, -1.5, -2])},
    }},
    "E3": {"title": "Refusal loop grids (test, 250 prompts x 3 CoTs x 3 answers): ablation + tuned actadd; "
                    "extra-loop recovery (Ouro 6 loops, Nanbeige 4, Dory 4)", "needs": ["E1", "E2"], "jobs": {
        "ouro4": {"hours": 2.5, "cmd": select("ouro", "grid_test", refusal_cands("ouro", L4))},
        "ouro6": {"hours": 3.5, "cmd": select("ouro", "grid_test6", refusal_cands("ouro", [[0, 1, 2, 3], None]), 6)},
        "nanbeige2": {"hours": 2.5, "cmd": select("nanbeige", "grid_test", refusal_cands("nanbeige", L2))},
        "nanbeige4": {"hours": 7.0, "cmd": select("nanbeige", "grid_test4", refusal_cands("nanbeige", [[0, 1], None]), 4)},
        "dory2": {"hours": 2.0, "cmd": select("dory", "grid_test", refusal_cands("dory", L2))},
        "dory4": {"hours": 5.0, "cmd": select("dory", "grid_test4", refusal_cands("dory", L4), 4)},
        "qwen": {"hours": 2.0, "cmd": select("qwen", "grid_test", qwen_refusal_cands())},
    }},
    "E4": {"title": "Dory weight orthogonalization at 2 and 4 loops (test, 250 prompts)", "needs": [], "jobs": {
        "dory2": {"hours": 1.5, "cmd": select("dory", "ortho_test", ORTHO)},
        "dory4": {"hours": 2.5, "cmd": select("dory", "ortho_test4", ORTHO, 4)},
    }},
    "E5": {"title": "Persona vectors: tune site and strength per trait for coherence (every loop / quarter at once)",
           "needs": [], "jobs": {k: {"hours": 3.0, "cmd": persona(k, "tune")} for k in M}},
    "E6": {"title": "Persona vectors in single loops / quarters and every loop; Dory 4 loops with loops 1+2",
           "needs": ["E5"], "jobs": {
        "ouro": {"hours": 1.5, "cmd": persona("ouro", "loops", ["1=0", "2=1", "3=2", "4=3", "all=0,1,2,3"])},
        "nanbeige": {"hours": 1.0, "cmd": persona("nanbeige", "loops", ["1=0", "2=1", "all=0,1"])},
        "dory2": {"hours": 1.0, "cmd": persona("dory", "loops", ["1=0", "2=1", "all=0,1"])},
        "dory4": {"hours": 2.0, "cmd": persona("dory", "loops", ["1=0", "2=1", "3=2", "4=3", "12=0,1", "all=0,1,2,3"],
                                               4)},
        "qwen": {"hours": 1.5, "cmd": persona("qwen", "loops", ["q1=0", "q2=1", "q3=2", "q4=3", "all=0,1,2,3"])},
    }},
    "E7": {"title": "Random kicks on MATH-500 (100 problems), a different random direction per problem, 4 sizes; "
                    "accuracy + teacher-forced KL", "needs": [], "jobs": {
        "ouro4": {"hours": 6.0, "cmd": kick_job("ouro", "kickp", [f"t{t}.l12" for t in range(4)])},
        "ouro6": {"hours": 3.5, "cmd": kick_job("ouro", "kickp_loops6", ["t3.l12", "t5.l12"], 6)},
        "nanbeige2": {"hours": 1.5, "cmd": kick_job("nanbeige", "kickp", ["t0.l11", "t1.l11"])},
        "nanbeige4": {"hours": 2.5, "cmd": kick_job("nanbeige", "kickp_loops4", ["t1.l11", "t3.l11"], 4)},
        "dory2": {"hours": 2.0, "cmd": kick_job("dory", "kickp", ["t0.l36", "t1.l36"])},
        "dory4": {"hours": 4.5, "cmd": kick_job("dory", "kickp_loops4", [f"t{t}.l36" for t in range(4)], 4)},
        "qwen": {"hours": 2.0, "cmd": kick_job("qwen", "kickp", ["l4", "l13", "l22", "l31"])},
    }},
    "E8": {"title": "Pilot: Dory (4 loops) kicked at every 4th sublayer of every loop (30 problems, eps 1) + KL at "
                    "every site", "needs": [], "jobs": {
        "dory4": {"hours": 3.5, "cmd": kick_job("dory", "kickp_pilot_loops4", DORY_PILOT_SITES, 4, eps="1", limit=30,
                                                kl_sites="all", kl_eps="0.5,1")},
    }},
}

OUTPUTS = {  # paths under data/runs pushed after each job (directories or files; missing ones are skipped)
    "E1": {"norms": EXPERIMENTS["E1"]["jobs"]["norms"]["outputs"]},
    "E2": {k: out(k, "generations/tune_actadd", "analysis/tune_actadd.csv", "capability/math500/tune_actadd",
                  "analysis/actadd_tune.json") for k in ["ouro", "nanbeige", "dory", "qwen"]},
    "E3": {"ouro4": out("ouro", "generations/grid_test", "analysis/grid_test.csv"),
           "ouro6": out("ouro", "generations/grid_test6", "analysis/grid_test6.csv"),
           "nanbeige2": out("nanbeige", "generations/grid_test", "analysis/grid_test.csv"),
           "nanbeige4": out("nanbeige", "generations/grid_test4", "analysis/grid_test4.csv"),
           "dory2": out("dory", "generations/grid_test", "analysis/grid_test.csv"),
           "dory4": out("dory", "generations/grid_test4", "analysis/grid_test4.csv"),
           "qwen": out("qwen", "generations/grid_test", "analysis/grid_test.csv")},
    "E4": {"dory2": out("dory", "generations/ortho_test", "analysis/ortho_test.csv"),
           "dory4": out("dory", "generations/ortho_test4", "analysis/ortho_test4.csv")},
    "E5": {k: out(k, "persona/steer_tune.parquet", "persona/steer_tune.judged.parquet", "analysis/persona_tune.csv",
                  "analysis/persona_tune.json") for k in M},
    "E6": {"ouro": out("ouro", "persona/steer_loops.parquet", "persona/steer_loops.judged.parquet",
                       "analysis/persona_loops.csv"),
           "nanbeige": out("nanbeige", "persona/steer_loops.parquet", "persona/steer_loops.judged.parquet",
                           "analysis/persona_loops.csv"),
           "dory2": out("dory", "persona/steer_loops.parquet", "persona/steer_loops.judged.parquet",
                        "analysis/persona_loops.csv"),
           "dory4": out("dory", "persona/steer_loops_loops4.parquet", "persona/steer_loops_loops4.judged.parquet",
                        "analysis/persona_loops_loops4.csv"),
           "qwen": out("qwen", "persona/steer_loops.parquet", "persona/steer_loops.judged.parquet",
                       "analysis/persona_loops.csv")},
    "E7": {"ouro4": out("ouro", "capability/math500/kickp", "analysis/kick_kl/kickp.parquet"),
           "ouro6": out("ouro", "capability/math500/kickp_loops6", "analysis/kick_kl/kickp_loops6.parquet"),
           "nanbeige2": out("nanbeige", "capability/math500/kickp", "analysis/kick_kl/kickp.parquet"),
           "nanbeige4": out("nanbeige", "capability/math500/kickp_loops4", "analysis/kick_kl/kickp_loops4.parquet"),
           "dory2": out("dory", "capability/math500/kickp", "analysis/kick_kl/kickp.parquet"),
           "dory4": out("dory", "capability/math500/kickp_loops4", "analysis/kick_kl/kickp_loops4.parquet"),
           "qwen": out("qwen", "capability/math500/kickp", "analysis/kick_kl/kickp.parquet")},
    "E8": {"dory4": out("dory", "capability/math500/kickp_pilot_loops4", "analysis/kick_kl/kickp_pilot_loops4.parquet")},
}
# E2 depends on E1's norms only through files; E3's coefficients come from E2 (analysis/actadd_tune.json per model)
NEEDS_FILES = {"E2": [f"{M[k][1]}/directions/stream_norms.pt" for k in M],
               "E3": [f"{M[k][1]}/analysis/actadd_tune.json" for k in M]
               + [f"{M['ouro'][1]}/directions/stream_norms_loops6.pt", f"{M['nanbeige'][1]}/directions/stream_norms_loops4.pt",
                  f"{M['dory'][1]}/directions/stream_norms_loops4.pt"],
               "E6": [f"{M[k][1]}/analysis/persona_tune.json" for k in M]}


def load_cloud_env(path=Path("/workspace/cloud-sync/env.sh")):
    """Take GCS_BUCKET, GCLOUD_ACCOUNT, CLOUDSDK_CONFIG and gcloud's PATH from the instance's cloud-sync env file when
    the shell did not source it (jobs started by a runner from a shell without it)."""
    if os.environ.get("GCS_BUCKET") or not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[7:].split("=", 1)
            os.environ[k.strip()] = os.path.expandvars(v.strip().strip('"').strip("'"))


def gcs_base():
    base = os.environ.get("LOOP_STEER_GCS") or (os.environ["GCS_BUCKET"].rstrip("/") + "/loop_steer"
                                                if os.environ.get("GCS_BUCKET") else None)
    return base


def gcloud(*args, check=True):
    cmd = ["gcloud"] + ([f"--account={os.environ['GCLOUD_ACCOUNT']}"] if os.environ.get("GCLOUD_ACCOUNT") else [])
    return subprocess.run(cmd + list(args), check=check)


def gcs_ready():
    if not gcs_base():
        print("GCS not configured (set GCS_BUCKET or LOOP_STEER_GCS; source /workspace/cloud-sync/env.sh)",
              file=sys.stderr)
        return False
    if subprocess.run(["bash", "-c", "command -v gcloud"], capture_output=True).returncode != 0:
        print("gcloud not on PATH (source /workspace/cloud-sync/env.sh)", file=sys.stderr)
        return False
    return True


def cmd_list(a):
    for e, x in EXPERIMENTS.items():
        hrs = sum(j["hours"] for j in x["jobs"].values())
        print(f"{e}: {x['title']}\n    needs {', '.join(x['needs']) or '-'}; about {hrs:.1f} GPU-hours; jobs: "
              + ", ".join(f"{j} ({v['hours']:g} h)" for j, v in x["jobs"].items()))


def jobs_of(e, only=None):
    x = EXPERIMENTS[e]
    names = only.split(",") if only else list(x["jobs"])
    bad = [n for n in names if n not in x["jobs"]]
    if bad:
        sys.exit(f"{e} has no job {bad}; jobs: {list(x['jobs'])}")
    return [(n, x["jobs"][n]) for n in names]


def cmd_show(a):
    for n, j in jobs_of(a.exp, a.jobs):
        print(f"## {a.exp}-{n} (~{j['hours']} h)\n{j['cmd']}\n   outputs: {OUTPUTS[a.exp][n]}\n")


def gpuq(*args, **kw):
    env = dict(os.environ, GPUQ_ROOT=str(QROOT))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "gpuq.py"), *args], env=env, cwd=ROOT, **kw)


def cmd_run(a):
    missing = [p for p in NEEDS_FILES.get(a.exp, []) if not (RUNS / p).exists()]
    if missing and gcs_ready():
        print(f"pulling dependencies {EXPERIMENTS[a.exp]['needs']} for {len(missing)} missing files")
        for dep in EXPERIMENTS[a.exp]["needs"]:
            pull(dep)
        missing = [p for p in NEEDS_FILES.get(a.exp, []) if not (RUNS / p).exists()]
    if missing and not a.force:
        sys.exit(f"{a.exp} needs files that are not here (run {EXPERIMENTS[a.exp]['needs']} first, or pull them):\n  "
                 + "\n  ".join(missing))
    gpus = a.gpus.split(",")
    load = {g: 0.0 for g in gpus}
    for n, j in sorted(jobs_of(a.exp, a.jobs), key=lambda nj: -nj[1]["hours"]):  # longest first, to the least loaded
        g = min(gpus, key=lambda x: load[x])
        load[g] += j["hours"]
        cmd = f"{j['cmd']} ; rc=$? ; {PY} scripts/exp.py push {a.exp} --job {n} ; exit $rc"
        gpuq("add", "-g", g, "-n", f"{a.exp}-{n}", "--idle", str(a.idle), "--", cmd, check=True)
        print(f"queued {a.exp}-{n} on GPU {g} (~{j['hours']} h)")
    for g in gpus:
        gpuq("start", "-g", g)
    print(f"estimated finish (hours from now, sequential per GPU): {load}\nqueue: GPUQ_ROOT={QROOT} "
          f"scripts/gpuq.py list -g N   (or scripts/exp.py status {a.exp})")


def cmd_status(a):
    for g in sorted(p.name[3:] for p in QROOT.glob("gpu*")):
        r = gpuq("list", "-g", g, capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if (a.exp or "E") in ln and "runner log" not in ln]
        if lines:
            print(f"GPU {g}:\n  " + "\n  ".join(lines))


def push(e, job=None):
    if not gcs_ready():
        print("push skipped: results stay in data/runs", file=sys.stderr)
        return
    dest = f"{gcs_base()}/results/{e}/{HOST}"
    paths = [p for n, ps in OUTPUTS[e].items() if job in (None, n) for p in ps]
    for p in dict.fromkeys(paths):
        src = RUNS / p
        if src.is_dir():
            gcloud("storage", "rsync", str(src), f"{dest}/{p}", "--recursive", check=False)
        elif src.exists():
            gcloud("storage", "cp", str(src), f"{dest}/{p}", check=False)
    logs = sorted((RUNS / "logs" / "gpuq").glob(f"*-{e}-{job or ''}*.log"))
    for f in logs:
        gcloud("storage", "cp", str(f), f"{dest}/logs/{f.name}", check=False)
    print(f"pushed {e}{'-' + job if job else ''} -> {dest}")


def pull(e):
    if not gcs_ready():
        sys.exit("pull needs GCS")
    base = f"{gcs_base()}/results/{e}"
    r = subprocess.run(["gcloud"] + ([f"--account={os.environ['GCLOUD_ACCOUNT']}"] if os.environ.get("GCLOUD_ACCOUNT")
                                     else []) + ["storage", "ls", base + "/"], capture_output=True, text=True)
    hosts = [ln.rstrip("/").rsplit("/", 1)[-1] for ln in r.stdout.split() if ln.endswith("/")]
    if not hosts:
        print(f"nothing under {base}/")
    for h in hosts:
        print(f"pulling {e} from {h}")
        gcloud("storage", "rsync", f"{base}/{h}", str(RUNS), "--recursive", "--no-clobber", check=False)


def cmd_push(a):
    push(a.exp, a.job)


def cmd_pull(a):
    for e in a.exps:
        pull(e)


def cmd_coef(a):
    path = RUNS / M[a.model][1] / "analysis" / "actadd_tune.json"
    if not path.exists():
        sys.exit(f"{path} missing: run E2 for {a.model} (scripts/exp.py run E2) or pull it (scripts/exp.py pull E2)")
    print(json.load(open(path))["c"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(f=cmd_list)
    p = sub.add_parser("show"); p.add_argument("exp", choices=EXPERIMENTS); p.add_argument("--jobs"); p.set_defaults(f=cmd_show)
    p = sub.add_parser("run"); p.add_argument("exp", choices=EXPERIMENTS); p.add_argument("--gpus", default="0")
    p.add_argument("--jobs", help="comma-separated subset"); p.add_argument("--idle", type=int, default=3600)
    p.add_argument("--force", action="store_true", help="queue even if dependency files are missing")
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("status"); p.add_argument("exp", nargs="?"); p.set_defaults(f=cmd_status)
    p = sub.add_parser("push"); p.add_argument("exp", choices=EXPERIMENTS); p.add_argument("--job"); p.set_defaults(f=cmd_push)
    p = sub.add_parser("pull"); p.add_argument("exps", nargs="+", choices=EXPERIMENTS); p.set_defaults(f=cmd_pull)
    p = sub.add_parser("coef"); p.add_argument("model", choices=M); p.set_defaults(f=cmd_coef)
    a = ap.parse_args()
    load_cloud_env()
    a.f(a)


if __name__ == "__main__":
    main()
