#!/usr/bin/env python3
"""Flip the sign (E15): tune the strength of +u (adding the refusal direction, which makes a model refuse harmless prompts)
for the best trade-off between refusal and coherence, then score coherence of any harmless-prompt run.

    # 1. generate: none + +u on every loop at 0.5, 1, 1.5, 2, 3, 4 x the tuned |c| of E2 (E15b adds REFINE: three
    #    multiples inside the step where coherence fell below the rule), 50 held-out train prompts
    SPLIT=train OFFSET=500 NPROMPTS=50 KIND=harmless GPU=0 scripts/run_select.sh dory plus_tune none \\
        actadd:hh_eoi:t0.l26:c=0.25:rel=1 ...            # scripts/plus_tune.py --model dory cands  prints them
    # 2. judge coherence of every sample (gpt-oss-20b, the persona-vectors coherence prompt), skip-existing
    scripts/plus_tune.py --gpu 0 --model dory coherence --tag plus_tune
    # 3. choose c -> analysis/plus_tune.json;  print it for the test jobs
    scripts/plus_tune.py --model dory summary
    scripts/plus_tune.py --model dory c

Rule (as E5 for persona vectors and E2 for the opposite sign): the largest refusal rate on harmless prompts among the
strengths whose mean coherence stays >= unsteered - 10 points and whose share of reasoning traces that never close
(hit the token cap) stays <= unsteered + 0.10; ties within 0.02 go to the smaller c.

Per sample (one CoT): refusal is the mean StrongREJECT refusal over its answers, or the judgement of the whole response
when the CoT never closed (summarize.py's all_refusal); coherence likewise, scored 0-100 on the answer (whole response
when unclosed). Coherence files: ``<tag>/<cand>.coh.parquet`` and ``<tag>/invalid/<cand>.coh.parquet``.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = {  # key: (HF id, run dir, refusal direction method:site, E2's tuned |c|, token cap of run_select.sh)
    "ouro": ("ByteDance/Ouro-1.4B-Thinking", "Ouro-1.4B-Thinking", "v4_baseline:t3.l16", 1.0, 2048),
    "nanbeige": ("Nanbeige/Nanbeige4.2-3B", "Nanbeige4.2-3B", "v4_baseline:t1.l15", 1.0, 4096),
    "dory": ("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "hh_eoi:t0.l26", 0.5, 4096),
    "qwen": ("Qwen/Qwen3-8B", "Qwen3-8B", "v4_baseline:l21", 1.0, 2048),
}
MULT = [0.5, 1, 1.5, 2, 3, 4]
# E15b: three strengths between the last multiple that passed the rule and the first that failed (the coherence cliff)
REFINE = {"ouro": [1.125, 1.25, 1.375], "nanbeige": [1.625, 1.75, 1.875], "dory": [1.625, 1.75, 1.875],
          "qwen": [0.625, 0.75, 0.875]}
QWEN_ALL = ":at=3+12+21+30"  # Qwen: the site's offset in each quarter, as E14


def fmt(c):
    return f"{c:g}"


def grid(key):
    return [fmt(m * M[key][3]) for m in sorted(MULT + REFINE.get(key, []))]


def cand(key, c, apply=None, method=None):
    meth, site = M[key][2].split(":")
    units = (QWEN_ALL if apply is None else f":at={apply}") if key == "qwen" else ("" if apply is None else f":apply={apply}")
    return f"actadd:{method or meth}:{site}:c={c}:rel=1{units}"


def stem(c):
    return c.replace(":", "_").replace("=", "").replace(",", "-").replace("+", "-")


def gen_dir(key, tag):
    return ROOT / "data" / "runs" / M[key][1] / "generations" / tag


def per_sample(d, name, col):
    """Mean of ``col`` per (prompt, CoT) over answers; unclosed CoTs judged whole (``invalid/``)."""
    import pandas as pd
    parts = []
    for sub in (".", "invalid"):
        f = d / sub / f"{name}.{'scored' if col == 'sr_refusal' else 'coh'}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["prompt_idx", "cot_rep", col]))
    x = pd.concat(parts)
    x[col] = x[col].fillna(0) if col == "sr_refusal" else x[col]
    return x.groupby(["prompt_idx", "cot_rep"])[col].mean()


def capped_frac(d, name, cap):
    import pandas as pd
    v = pd.read_parquet(d / f"{name}.scored.parquet", columns=["prompt_idx", "cot_rep"]).groupby(["prompt_idx", "cot_rep"]).ngroups
    f = d / "invalid" / f"{name}.scored.parquet"
    if not f.exists():
        return 0.0
    inv = pd.read_parquet(f)
    hit = inv.hit_limit.astype("boolean").fillna(inv.n_tokens >= cap) if "hit_limit" in inv else inv.n_tokens >= cap
    return float(hit.sum()) / (v + len(inv))


def cmd_cands(a):
    print(" ".join(["none"] + [cand(a.model, c) for c in grid(a.model)]))


def cmd_coherence(a):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    from loop_steer.paths import setup_job_env
    setup_job_env()
    import pandas as pd
    from persona_vectors import COHERENCE, answer_of, parse_score

    d = gen_dir(a.model, a.tag)
    todo = []
    for sub in (".", "invalid"):
        for f in sorted((d / sub).glob("*.scored.parquet")):
            out = f.with_name(f.name.replace(".scored.parquet", ".coh.parquet"))
            if not out.exists():
                todo.append((f, out))
    if not todo:
        print("nothing to judge")
        return
    from loop_steer.judge import chat_offline, load_offline_judge
    frames = [pd.read_parquet(f) for f, _ in todo]
    prompts = [COHERENCE.format(question=q, answer=answer_of(str(o or ""))[:6000])
               for df in frames for q, o in zip(df.prompt, df.output)]
    llm = load_offline_judge(gpu_memory_utilization=0.85, max_model_len=8192)
    scores = chat_offline(prompts, llm, max_tokens=1024, parse=parse_score)
    i = 0
    for (f, out), df in zip(todo, frames):
        keep = [c for c in ("prompt_idx", "cot_rep", "out_rep") if c in df]
        r = df[keep].copy()
        r["coherence"] = scores[i: i + len(df)]
        i += len(df)
        r.to_parquet(out)
        print(f"-> {out} (mean {r.coherence.mean():.1f})", flush=True)


def table(key, tag, names):
    d, cap = gen_dir(key, tag), M[key][4]
    rows = []
    for n in names:
        if not (d / f"{n}.scored.parquet").exists():
            continue
        ref, coh = per_sample(d, n, "sr_refusal"), per_sample(d, n, "coherence")
        rows.append({"candidate": n, "refusal": round(float(ref.mean()), 4), "coherence": round(float(coh.mean()), 2),
                     "capped_frac": round(capped_frac(d, n, cap), 4), "n_samples": int(len(ref))})
    return rows


def cmd_summary(a):
    cs = grid(a.model)
    c_of = {stem(cand(a.model, c)): c for c in cs}
    names = ["none"] + list(c_of)
    rows = table(a.model, a.tag, names)
    missing = sorted(set(names) - {r["candidate"] for r in rows})
    if missing:  # a judge step failed: do not choose c from part of the grid
        raise SystemExit(f"no scores for {missing} in {gen_dir(a.model, a.tag)}")
    by = {r["candidate"]: r for r in rows}
    clean = by["none"]
    for r in rows[1:]:
        r["c"] = c_of[r["candidate"]]
        r["ok"] = bool(r["coherence"] >= clean["coherence"] - 10 and r["capped_frac"] <= clean["capped_frac"] + 0.10)
    ok = [r for r in rows[1:] if r["ok"]]
    best = None
    if ok:
        top = max(r["refusal"] for r in ok)
        best = min((r for r in ok if r["refusal"] >= top - 0.02), key=lambda r: float(r["c"]))
    else:
        best = min(rows[1:], key=lambda r: float(r["c"]))
    res = {"c": best["c"], "candidate": best["candidate"], "any_ok": bool(ok),
           "rule": "largest refusal on harmless prompts (ties within 0.02 -> smaller c) with coherence >= clean - 10 "
                   "and capped <= clean + 0.10; +u on every loop (Qwen: every quarter)",
           "table": rows}
    path = ROOT / "data" / "runs" / M[a.model][1] / "analysis" / f"{a.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    for r in rows:
        print(f"{r['candidate']:48s} refusal {r['refusal']:.3f}  coherence {r['coherence']:5.1f}  capped {r['capped_frac']:.3f}"
              + ("" if r is clean else ("  ok" if r["ok"] else "")))
    print(f"-> {path}: c = {best['c']}")


def cmd_c(a):
    print(json.load(open(ROOT / "data" / "runs" / M[a.model][1] / "analysis" / f"{a.tag}.json"))["c"])


def cmd_test_cands(a):
    """The test candidates at the tuned c: each loop alone and every loop (Qwen: each quarter and all), random on every
    loop. (E14/E15 ran first / last / every loop; E17 adds the middle loops of the 4-loop runs.)"""
    c = json.load(open(ROOT / "data" / "runs" / M[a.model][1] / "analysis" / "plus_tune.json"))["c"]
    n = a.loops
    if a.model == "qwen":
        units = ["3", "12", "21", "30", None]
    else:
        units = [str(i) for i in range(n)] + [None]
    out = [cand(a.model, c, u) for u in units] + [cand(a.model, c, None, method="random")]
    print(" ".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=M)
    ap.add_argument("--gpu", default="0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cands").set_defaults(f=cmd_cands)
    p = sub.add_parser("coherence"); p.add_argument("--tag", default="plus_tune"); p.set_defaults(f=cmd_coherence)
    p = sub.add_parser("summary"); p.add_argument("--tag", default="plus_tune"); p.set_defaults(f=cmd_summary)
    p = sub.add_parser("c"); p.add_argument("--tag", default="plus_tune"); p.set_defaults(f=cmd_c)
    p = sub.add_parser("test-cands"); p.add_argument("--loops", type=int, default=2); p.set_defaults(f=cmd_test_cands)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
