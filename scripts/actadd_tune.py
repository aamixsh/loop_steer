#!/usr/bin/env python3
"""Pick the activation-addition coefficient per model from the tuning run (experiment E2, ``scripts/exp.py``).

    scripts/actadd_tune.py --model ouro|nanbeige|dory|qwen [--tag tune_actadd]

Reads the selection-set summary ``analysis/<tag>.csv`` (``scripts/run_select.sh``: all_refusal, capped_frac) and the
MATH-500 runs ``capability/math500/<tag>/*.parquet`` of the same candidates. Rule: among the actadd candidates whose
MATH-500 accuracy is at least clean - ``--max-acc-drop`` (default 5 points; 100 problems give a standard error of
about 4) and whose capped / unclosed CoT share is at most clean + ``--max-capped-rise`` (default 0.10), take the one
with the lowest all_refusal; ties (within 0.02) go to the smaller |c|. Writes ``analysis/actadd_tune.json``
(``{"c": ..., "candidate": ..., "table": [...]}``), which E3 reads (``scripts/exp.py coef MODEL``). CPU only.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"ouro": "Ouro-1.4B-Thinking", "nanbeige": "Nanbeige4.2-3B", "dory": "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
        "qwen": "Qwen3-8B"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(RUNS))
    ap.add_argument("--tag", default="tune_actadd")
    ap.add_argument("--max-acc-drop", type=float, default=0.05)
    ap.add_argument("--max-capped-rise", type=float, default=0.10)
    args = ap.parse_args()
    rd = ROOT / "data" / "runs" / RUNS[args.model]
    sel = pd.read_csv(rd / "analysis" / f"{args.tag}.csv")
    acc = {}
    for f in sorted((rd / "capability" / "math500" / args.tag).glob("*.parquet")):
        df = pd.read_parquet(f)
        acc[f.stem] = (float(df.correct.mean()), float(df.closed.mean()), len(df))
    rows = []
    for r in sel.itertuples():
        m = re.search(r"_c(-?[\d.]+)", r.candidate)
        a = acc.get(r.candidate, (float("nan"), float("nan"), 0))
        rows.append({"candidate": r.candidate, "c": float(m.group(1)) if m else 0.0, "all_refusal": r.all_refusal,
                     "capped_frac": r.capped_frac if pd.notna(r.capped_frac) else 0.0, "math500_acc": a[0],
                     "math500_closed": a[1], "math500_n": a[2]})
    t = pd.DataFrame(rows).sort_values("c", ascending=False)
    clean = t[t.candidate == "none"].iloc[0]
    ok = t[(t.candidate != "none") & (t.math500_acc >= clean.math500_acc - args.max_acc_drop)
           & (t.capped_frac <= clean.capped_frac + args.max_capped_rise)]
    print(t.round(3).to_string(index=False))
    if not len(ok):
        raise SystemExit("no candidate passes the capability / capping limits; inspect the table and choose by hand "
                         f"(write {rd / 'analysis' / 'actadd_tune.json'} with a 'c' key)")
    best = ok[ok.all_refusal <= ok.all_refusal.min() + 0.02].sort_values("c", ascending=False).iloc[0]
    out = {"c": float(best.c), "candidate": best.candidate,
           "rule": f"lowest all_refusal (ties within 0.02 -> smaller |c|) with MATH-500 acc >= clean - "
                   f"{args.max_acc_drop} and capped <= clean + {args.max_capped_rise}",
           "table": t.to_dict(orient="records")}
    path = rd / "analysis" / "actadd_tune.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"chosen c = {best.c} ({best.candidate}) -> {path}")


if __name__ == "__main__":
    main()
